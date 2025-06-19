'''32638549804688
decoder time: 0.1
by lyuwenyu
'''
# import time 
# import json
# import datetime
# from copy import deepcopy
# import torch 
# import csv
# import os
# from src.misc import dist
# from src.data import get_coco_api_from_dataset

# from .solver import BaseSolver
# from .det_engine import train_one_epoch, evaluate


# class DetSolver(BaseSolver):
    
#     def fit(self, ):
#         print("Start training")
#         self.train()

#         args = self.cfg 
        
#         n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         print('number of params:', n_parameters)
#         base_ds = get_coco_api_from_dataset(deepcopy(self.val_dataloader.dataset))
#         coco_map = [[None], ['AP50-95', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl', 'Acc1']]
#         best_acc1 = 0
#         best_stat = {'epoch': -1, }

#         start_time = time.time()
#         for epoch in range(self.last_epoch + 1, args.epoches):
#             if dist.is_dist_available_and_initialized():
#                 self.train_dataloader.sampler.set_epoch(epoch)

#             train_stats = train_one_epoch(
#                 self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
#                 args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

#             self.lr_scheduler.step()

#             if self.output_dir:
#                 checkpoint_paths = [self.output_dir / 'checkpoint.pth']
#                 if (epoch + 1) % args.checkpoint_step == 0:
#                     checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
#                 for checkpoint_path in checkpoint_paths:
#                     dist.save_on_master(self.state_dict(epoch), checkpoint_path)

#             module = self.ema.module if self.ema else self.model
#             test_stats, coco_evaluator = evaluate(
#                 module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
#             )

#             if test_stats['acc1'] > best_acc1:
#                 best_acc1 = test_stats['acc1']


#             # TODO 
#             for k in test_stats.keys():
#                 if k in best_stat:
#                     best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
#                     best_stat[k] = max(best_stat[k], test_stats[k][0])
#                 elif k != 'acc1':
#                     best_stat['epoch'] = epoch
#                     best_stat[k] = test_stats[k][0]
#             print('best_stat: ', best_stat, '\nbest_acc1: ', best_acc1)


#             coco_map.append(test_stats['coco_eval_bbox'] + [test_stats['acc1']])
#             coco_map[0] = [best_stat['epoch'], best_stat['coco_eval_bbox'], str(best_acc1)]
#             with open(os.path.join(self.output_dir, 'coco_map.csv'), 'w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerows(coco_map)

#             log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
#                         **{f'test_{k}': v for k, v in test_stats.items()},
#                         'epoch': epoch,
#                         'n_parameters': n_parameters}

#             if self.output_dir and dist.is_main_process():
#                 with (self.output_dir / "log.txt").open("a") as f:
#                     f.write(json.dumps(log_stats) + "\n")

#                 # for evaluation logs
#                 if coco_evaluator is not None:
#                     (self.output_dir / 'eval').mkdir(exist_ok=True)
#                     if "bbox" in coco_evaluator.coco_eval:
#                         filenames = ['latest.pth']
#                         if epoch % 50 == 0:
#                             filenames.append(f'{epoch:03}.pth')
#                         for name in filenames:
#                             torch.save(coco_evaluator.coco_eval["bbox"].eval,
#                                     self.output_dir / "eval" / name)

#         total_time = time.time() - start_time
#         total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#         print('Training time {}'.format(total_time_str))


#     def val(self, ):
#         self.eval()
#         base_ds = get_coco_api_from_dataset(deepcopy(self.val_dataloader.dataset))

#         module = self.ema.module if self.ema else self.model
#         test_stats, coco_evaluator = evaluate(
#             module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
#         )
#         print('mAP50-95:', test_stats['coco_eval_bbox'][0])

#         return

import time 
import json
import datetime
from copy import deepcopy
import torch 
import csv
import os
from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        base_ds = get_coco_api_from_dataset(deepcopy(self.val_dataloader.dataset))
        coco_map = [[None], ['AP50-95', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'AR1', 'AR10', 'AR100', 'ARs', 'ARm', 'ARl', 'Acc1']]
        best_acc1 = 0
        best_map = 0  # Track best mAP
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # Check if this is the best model so far (based on mAP)
            current_map = test_stats['coco_eval_bbox'][0]
            is_best = current_map > best_map
            
            if is_best:
                best_map = current_map
                # Only save checkpoint for the best model
                if self.output_dir:
                    best_checkpoint_path = self.output_dir / 'best_model.pth'
                    dist.save_on_master(self.state_dict(epoch), best_checkpoint_path)
                    print(f"New best model saved at epoch {epoch} with mAP: {best_map:.4f}")

            if test_stats['acc1'] > best_acc1:
                best_acc1 = test_stats['acc1']

            # Update best stats
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                elif k != 'acc1':
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            print('best_stat: ', best_stat, '\nbest_acc1: ', best_acc1)

            coco_map.append(test_stats['coco_eval_bbox'] + [test_stats['acc1']])
            coco_map[0] = [best_stat['epoch'], best_stat['coco_eval_bbox'], str(best_acc1)]
            with open(os.path.join(self.output_dir, 'coco_map.csv'), 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(coco_map)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        print(f"Best model saved with mAP: {best_map:.4f} at epoch {best_stat['epoch']}")


    def val(self, ):
        self.eval()
        base_ds = get_coco_api_from_dataset(deepcopy(self.val_dataloader.dataset))

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
        )
        print('mAP50-95:', test_stats['coco_eval_bbox'][0])

        return