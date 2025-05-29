import argparse
import os
try:
    import ruamel_yaml as yaml
except:
    import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from models.model_video_caption_mplug import MPLUG2
from models.vit import interpolate_pos_embed, resize_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import save_result
from dataset import create_dataset, create_sampler, create_loader, vqa_collate_fn

from scheduler import create_scheduler
from optim import create_optimizer, create_two_optimizer

# import language_evaluation

import warnings
warnings.filterwarnings("ignore")


def test_collect_fn(batch):
    video_list, video_id_list, golden_captions = [], [], []
    for video, video_id, caption in batch:
        video_list.append(video)
        video_id_list.append(video_id)
        golden_captions.append(caption)
    return torch.stack(video_list, dim=0), video_id_list, golden_captions


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config, do_amp=False,
          do_two_optim=False, do_accum=True, accum_steps=1):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if do_two_optim:
        metric_logger.add_meter('lr1', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
        metric_logger.add_meter('lr2', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    else:
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    len_batch = len(data_loader)
    print("Total Batch {}".format(len_batch))

    for i, (video, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        video = video.to(device,non_blocking=True)
        if config['prompt'] != "":
            caption = [config['prompt'] + each + config['eos'] for each in caption]
        else:
            caption = [each + config['eos'] for each in caption]
        # question_input = [""] # tokeninzer would add [CLS] automatically
        
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=config['max_length'], return_tensors="pt").to(device)
        # question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=config['max_length'], return_tensors="pt").to(device)
        
        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss = model(video, caption, train=True, alpha=alpha)
        if accum_steps > 1:
            loss = loss / accum_steps

        if do_amp:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                # logger.info('scaled loss: {}'.format(str(scaled_loss)))
                scaled_loss.backward()
        else:
            loss.backward()
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        metric_logger.update(loss=loss.item())

        if do_two_optim:
            metric_logger.update(lr1=optimizer.param_groups[0]["lr"])
            metric_logger.update(lr2=optimizer.param_groups[2]["lr"])
        else:
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        elif scheduler.step_mode:
            scheduler.step(epoch * len_batch + i)
        

            # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Generate Caption test result:'
    print_freq = 50

    result = []

    answer_input = None
    number_1 = 0
    number_2 = 0
    for n, (videos, video_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(video.shape, video_ids)
        # exit(0)
        temp_path = os.path.join("/home/jupyter/snapdataset/1013_en/features/caption_snap1/", video_ids[0] + ".npy")
        if os.path.exists(temp_path):
            number_1 = number_1 + 1
            continue
        try:
            b, c, n, h, w = videos.shape
            n_seq = 16
            n_seq_multiple = n // n_seq
            n_seq_residual = n % n_seq
            videos = videos.cuda()

            if n_seq_residual > 3:
                videos = torch.cat((videos[:,:,0:n_seq_multiple*n_seq], videos[:,:,-n_seq:]), dim=2)
            elif n_seq_residual != 0:
                videos = videos[:,:,0:n_seq_multiple*n_seq]
            # print(videos.shape)
            # exit(0)
            video_new_len = videos.shape[2]
            videos = videos.permute(0,2,1,3,4)[0].view(video_new_len//n_seq, n_seq, c, h, w)
            # print(videos.shape)
            # exit(0)
            image_embeds_list = []
            bs = 4
            for kkk in range(video_new_len // n_seq // bs):
                # video = videos[:,:,kkk*n_seq*4:(kkk+1)*n_seq*4]
                # print(video.shape)
                # exit(0)
                video = videos[kkk*bs:(kkk+1)*bs,:].permute(0,2,1,3,4)
                # video = video.to(device,non_blocking=True)
                # topk_ids, topk_probs, image_embeds = model(video, train=False)
                image_embeds = model(video, train=False)
                image_embeds_list.append(image_embeds)
            if (kkk+1)* bs < video_new_len // n_seq:
                video = videos[(kkk+1)*bs:,:].permute(0,2,1,3,4)
                image_embeds = model(video, train=False)
                image_embeds_list.append(image_embeds)
                # print(image_embeds.shape)
                # exit(0)
                # for video_id, topk_id, topk_prob, gold_caption_list in zip(video_ids, topk_ids, topk_probs, gold_caption):
                #    ans = tokenizer.decode(topk_id[0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
                #    if config["prompt"] != "":
                #        ans = ans.split(config["prompt"])[-1].strip()
                #    # if n % 50 == 0:
                #    print(video_id, ans)
                #    result.append({"video_id": video_id, "pred_caption":ans, "gold_caption": gold_caption_list})
            
            image_embeds_all = torch.cat(image_embeds_list, dim=0)
            image_embeds_all = torch.mean(image_embeds_all, dim=1)
        except:
            number_2 = number_2 + 1
            print("failed", n, video_ids)
            continue
        # print(image_embeds_all.shape, video_ids)
        # exit(0)
        image_embeds_all_np = image_embeds_all.cpu().data.numpy()
        # print(image_embeds_all_np.dtype, image_embeds_all_np.shape, b, c, n, h, w)
        np.save(os.path.join("/home/jupyter/snapdataset/1013_en/features/caption_snap1/", video_ids[0] + ".npy"), image_embeds_all_np)
        # exit(0)
        if n == 0:
            print(result)
    print(number_1, number_2)
    exit(0)
    return result

@torch.no_grad()
def evaluate(model, data_loader, dataset, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    # exit(0)
    print_freq = 50
    predicts = []
    answers = []
    answer_input = None
    for n, (video, video_ids, gold_caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):  
        video = video.to(device,non_blocking=True)             
        caption = [each+config['eos'] for each in caption]
        question_input = [config['bos']]*len(caption)
        caption = tokenizer(caption, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)
        question_input = tokenizer(question_input, padding='longest', truncation=True, max_length=args.max_input_length, return_tensors="pt").to(device)

        for i in range(len(gold_caption)):
            predicts.append(gold_caption[i][0])
            answers.append(gold_caption[i])
        #{'Bleu_1': 0.9999999999863945, 'Bleu_2': 0.9999999999859791, 'Bleu_3': 0.9999999999854866, 'Bleu_4': 0.999999999984889, 'METEOR': 1.0, 'ROUGE_L': 1.0, 'CIDEr': 2.7246232035629268, 'SPICE': 0.40389416048620613}
        result = cal_metric(predicts, answers)
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=video.size(0))
        metric_logger.meters['Bleu_2'].update(result["Bleu_1"], n=video.size(0))
        metric_logger.meters['Bleu_3'].update(result["Bleu_1"], n=video.size(0))
        metric_logger.meters['Bleu_4'].update(result["Bleu_1"], n=video.size(0))
        metric_logger.meters['Bleu_1'].update(result["Bleu_1"], n=video.size(0))

    # gather the stats from all processes
    torch.cuda.empty_cache()
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    for each in result_list:
        predicts.append(each["pred_caption"])
        answers.append(each["gold_caption"])
    evaluator = language_evaluation.CocoEvaluator(verbose=False)
    results = evaluator.run_evaluation(predicts, answers)
    print (len(result_list), results)
    return results

def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    print("Creating video caption datasets")
    #if args.no_randaug:
    #    datasets = create_dataset('video_caption_no_randaug', config)
    #else:
    #    datasets = create_dataset('video_caption', config)
    datasets = create_dataset('video_caption2', config)
    # exit(0)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False], num_tasks, global_rank)         
    else:
        samplers = [None, None]

    train_loader, val_loader = create_loader(datasets,samplers,
                                            batch_size=[config['batch_size_train'],config['batch_size_test']],
                                            num_workers=[4, 4],is_trains=[True, False], 
                                            collate_fns=[None, test_collect_fn]) 


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = MPLUG2(config=config, tokenizer=tokenizer)
    model = model.to(device)

    if not args.do_two_optim:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model)
    else:
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_two_optimizer(arg_opt, model)

    arg_sche = utils.AttrDict(config['schedular'])
    train_step_per_epoch = len(train_loader)
    print("train_step_per_epoch: {}".format(train_step_per_epoch))
    arg_sche["num_iterations"] = max_epoch * train_step_per_epoch - arg_sche['warmup_epochs']
    # lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    #if args.do_amp:
    #    from apex import amp
    #    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['module']

        # reshape positional embedding to accomodate for image resolution change
        if config["clip_name"] == "ViT-B-16":
            num_patches = int(config["image_res"] * config["image_res"]/(16*16))
        elif config["clip_name"] == "ViT-L-14":
            num_patches = int(config["image_res"] * config["image_res"]/(14*14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())

        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                                   pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

        if not args.evaluate:
            for key in list(state_dict.keys()):
                if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                    encoder_key = key.replace('fusion.', '').replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]


        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model_without_ddp = model
    # if args.distributed:
    #    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    #    import apex
    #    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    #     model_without_ddp = model.module

    best_epoch = -1
    best_acc = 0
    print("Start training")
    start_time = time.time()
    # exit(0)
    caption_result = evaluation(model, val_loader, tokenizer, device, config)
    exit(0)
    result_file = save_result(caption_result, args.result_dir, 'caption_result_zeroshot')
    if utils.is_main_process():
        result = cal_metric(result_file)
        val_stats = result

    if utils.is_main_process():
        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': -1,
                        }
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
        best_acc = float(val_stats['CIDEr'])       

    dist.barrier()
    # exit(0)
    for epoch in range(start_epoch, max_epoch):
        # if epoch > 0:
        #     lr_scheduler.step(epoch + warmup_steps)

            
        if not args.evaluate:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config, do_amp=args.do_amp, do_two_optim=args.do_two_optim, accum_steps=args.accum_steps)

        if args.evaluate:
            break

        caption_result = evaluation(model, val_loader, tokenizer, device, config)
        result_file = save_result(caption_result, args.result_dir, 'caption_result_epoch%d'%epoch)
        if utils.is_main_process():       
            result = cal_metric(result_file)
            val_stats = result
            
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'val_{k}': v for k, v in val_stats.items()},
                         'epoch': epoch,
                        }                
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")                        
                         
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))

            if float(val_stats['CIDEr']) >= best_acc:
                best_epoch = epoch
                best_acc = float(val_stats['CIDEr'])
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))

        if not lr_scheduler.step_mode:
            lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()
        torch.cuda.empty_cache()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        if not args.evaluate:
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write("best epoch: %d\n"%best_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/VQA.yaml')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--output_dir', default='output/vqa')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--text_decoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--min_length', default=4, type=int)
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--max_length', default=20, type=int)
    parser.add_argument('--max_input_length', default=25, type=int)
    parser.add_argument('--beam_size', default=5, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--do_two_optim', action='store_true')
    parser.add_argument('--add_object', action='store_true')
    parser.add_argument('--do_amp', action='store_true')
    parser.add_argument('--no_init_decocde', action='store_true')
    parser.add_argument('--do_accum', action='store_true')
    parser.add_argument('--no_prompt', action='store_true')
    parser.add_argument('--no_randaug', action='store_true')
    parser.add_argument('--accum_steps', default=2, type=int)

     # Model architecture
    parser.add_argument('--temporal_stride', default=2, type=int)
    parser.add_argument('--temporal_downsampling', action='store_true')

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    args.result_dir = os.path.join(args.output_dir, 'result')

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    config["min_length"] = args.min_length
    config["max_length"] = args.max_length
    config["add_object"] = args.add_object
    config["beam_size"] = args.beam_size
    #config['optimizer']['lr'] = args.lr
    #config['schedular']['lr'] = args.lr
    config['text_encoder'] = args.text_encoder
    config['text_decoder'] = args.text_decoder

    config['temporal_stride'] = args.temporal_stride
    config['temporal_downsampling'] = args.temporal_downsampling
    config['accum_steps'] = args.accum_steps
    config['no_randaug'] = args.no_randaug
    if args.no_prompt:
        config["prompt"] = ""

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)