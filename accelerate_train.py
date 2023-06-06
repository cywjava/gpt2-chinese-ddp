# coding=UTF-8
import argparse
import os
from glob import glob

import torch
import transformers
from accelerate import Accelerator
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Config

from GPT2DataSet import GPT2DataSet
from tokenizations import tokenization_jieba as tokenizer

"""
--分布式训练 
transformers==4.28.1
nohup accelerate launch --config_file /home/public/.cache/huggingface/accelerate/default_config.yaml  ./accelerate_train.py --model_config /home/public/train-json/bids_book_technology/model_config.json --tokenizer_path /home/pu
blic/train-json/bids_book_technology/jieba_ext_dict.txt --raw_data_path /home/public/train-json/bids_book_technology/0.json --epochs 200 --output_dir /home/public/train-json/bids_book_technology/model/ --batch_size 4 --fp16 >> 
./logs/acc.log 2>&1 &
"""

full_tokenizer: tokenizer.JiebaTokenizer = None


def start_train(finetune_args):
    global full_tokenizer
    if finetune_args.debug:
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["TORCH_DISTRIBUTED_DETAIL"] = "DEBUG"
    accelerator = Accelerator(gradient_accumulation_steps=finetune_args.gradient_accumulation_steps,
                              mixed_precision="fp16" if finetune_args.fp16 else "no")
    model_config = GPT2Config.from_json_file(finetune_args.model_config)
    full_tokenizer = tokenizer.JiebaTokenizer(ext_vocab_file=finetune_args.tokenizer_path)
    if model_config.vocab_size <= 0:
        model_config.vocab_size = full_tokenizer.get_vocab_size()
    accelerator.print("*" * 100)
    accelerator.print("using model_config:\n", model_config)
    model = GPT2LMHeadModel(config=model_config)
    torch.cuda.empty_cache()
    if not os.path.exists(finetune_args.output_dir):
        os.mkdir(finetune_args.output_dir)

    if accelerator.is_main_process:
        num_parameters = 0
        parameters = model.parameters()
        for parameter in parameters:
            num_parameters += parameter.numel()
        accelerator.print('number of parameters: {}'.format(num_parameters))

    # load dataset
    train_dataset = GPT2DataSet(GPT2DataSet.load_json(glob(pathname=finetune_args.raw_data_path)), tokenizer)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, seed=42)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=finetune_args.batch_size,
                                                    shuffle=(train_sampler is None),
                                                    sampler=train_sampler, drop_last=False,
                                                    collate_fn=collate_fn)
    # 定义优化器和学习率调整策略
    optimizer = transformers.AdamW(model.parameters(), lr=finetune_args.learning_rate, correct_bias=True)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.9999)
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_data_loader, lr_scheduler)

    accelerator.print('starting training')
    overall_step = 0
    single_epoch_steps = len(train_data_loader)
    for epoch in tqdm(range(0, finetune_args.epochs), desc="\nOverall progress", colour="GREEN",
                      unit="epoch", disable=not accelerator.is_main_process):
        model.train()
        train_sampler.set_epoch(epoch)
        with tqdm(range(single_epoch_steps), desc="Epoch " + str(epoch + 1) + " progress", colour="GREEN", unit="step",
                  disable=not accelerator.is_main_process) as epoch_process_bar:
            for step, batch in enumerate(train_data_loader):
                accelerator.free_memory()
                with accelerator.accumulate(model):
                    batch_inputs = []
                    for ids in batch.get("input_ids"):
                        batch_inputs.append(int(ids))
                    batch_inputs = torch.tensor(batch_inputs).long()
                    outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
                    loss, logits = outputs[:2]
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    overall_step += 1
                    epoch_process_bar.update(1)
            accelerator.print(f"\nstep:{overall_step},loss:{loss}")
            save_model(model, accelerator, finetune_args, (epoch + 1))


def save_model(_model, _accelerator, _finetune_args, _epoch):
    unwrapped_model = _accelerator.unwrap_model(_model)
    model_to_save = unwrapped_model.module if hasattr(unwrapped_model, 'module') else unwrapped_model
    model_to_save.save_pretrained(_finetune_args.output_dir + os.sep + 'epoch_' + str(_epoch))


def collate_fn(features: list) -> dict:
    sublines = [full_tokenizer.tokenize(line, sub_tokenization=True) for line in features if len(line) > 1]
    sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
    full_line = []
    for subline in sublines:
        full_line.append(full_tokenizer.convert_tokens_to_ids(['[MASK]'])[0])  # 文章开头添加MASK表示文章开始
        full_line.extend(subline)
        full_line.append(full_tokenizer.convert_tokens_to_ids(['[CLS]'])[0])  # 文章之间添加CLS表示文章结束

    return {
        "input_ids": full_line,
        "label_ids:": full_line
    }


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False,
                        help='模型参数配置文件')
    parser.add_argument('--tokenizer_path', default='cache/jieba_ext_dict.txt', type=str, required=False,
                        help='词表文件')
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False, help='原始训练语料')
    parser.add_argument('--output_dir', default="../output_dir", type=str, required=False, help='模型输出目录')
    parser.add_argument('--epochs', default=5, type=int, required=False, help='训练轮次')
    parser.add_argument('--learning_rate', default=1e-4, type=float, required=False, help='学习率')
    parser.add_argument('--batch_size', default="4", type=int, required=False, help='训练批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help="梯度累积步长")
    parser.add_argument('--debug', action='store_true', help='Whether print nccl & torch distributed detail info')
    parser.add_argument('--fp16', action='store_true', help="是否启用fp16")

    return parser.parse_args()


if __name__ == '__main__':
    start_train(set_args())
