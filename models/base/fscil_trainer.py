# from black import schedule_formatting
from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy

from .helper import *
from utils import *
from dataloader.data_utils import *
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch.backends.cudnn as cudnn

import util.misc as misc
from torch.utils.tensorboard.writer import SummaryWriter



class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self.set_seed()
        self.set_save_path()
        self.args = set_up_datasets(self.args)
        self.model = self.create_model()
        self.teacher_model = self.create_model()
        self.calculate_model_complexity()

    def get_optimizer_base(self):
        param_groups = [
            {'params': self.model.parameters(), 'lr': self.args.lr_base}
        ]
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), weight_decay=0.05)

        return optimizer

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        pass

    def set_save_path(self):
        mode = self.args.base_mode + '-' + self.args.new_mode
        if not self.args.not_data_init:
            mode = mode + '-' + 'data_init'

        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project
        self.args.save_path = self.args.save_path + '%s-start_%d/' % (mode, self.args.start_session)
        if 'cos' in mode:
            self.args.save_path = self.args.save_path + '-T_%.2f' % (self.args.temperature)
        if 'ft' in self.args.new_mode:
            self.args.save_path = self.args.save_path + '-ftLR_%.3f-ftEpoch_%d' % (
                self.args.lr_new, self.args.epochs_new)
        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)
        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)


        #tensorboard path
        self.log_writer = SummaryWriter(log_dir=self.args.save_path)



        return None









        # self.model.load_state_dict(self.best_model_dict)
        
        # train_set, trainloader, testloader = self.get_dataloader(0)
        # print('new classes for this session:\n', np.unique(train_set.targets))
        # optimizer = self.get_optimizer_base()
        # loss_scaler = NativeScaler()
        # torch.set_float32_matmul_precision('high')
        # #训练
        # for epoch in range(args.epochs_base):
            
        #     start_time = time.time()
        #     # train base sess
        #     tl, ta = base_train(self.model, trainloader, optimizer, epoch, loss_scaler, self.log_writer,  args)
            
            
        #     if epoch%50 == 0:
        #         save_model_dir = os.path.join(args.save_path, str(epoch) + '.pth')
        #         torch.save(dict(params=self.model.state_dict()), save_model_dir)
        #         torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
        #         self.best_model_dict = deepcopy(self.model.state_dict())
        #         print('********A model is found!!**********')
        #         print('Saving model to :%s' % save_model_dir)


        #     self.trlog['train_loss'].append(tl)
            
        #     # lrc = scheduler.get_last_lr()[0]
        #     lrc = optimizer.param_groups[0]['lr']
        #     # print(lrc)
        #     # print(tl)
        #     result_list.append(
        #         'epoch:%03d,lr:%.4f,training_loss:%.5f' % (
        #             epoch, lrc, tl))
        #     print('This epoch takes %d seconds' % (time.time() - start_time),
        #             '\nstill need around %.2f mins to finish this session' % (
        #                     (time.time() - start_time) * (args.epochs_base - epoch) / 60))
        #     # scheduler.step()

    def set_seed(self):
        # fix the seed for reproducibility
        seed = self.args.seed + misc.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        cudnn.benchmark = True

    def create_model(self):
        model = MYNET(self.args, mode=self.args.base_mode)
        model = nn.DataParallel(model, list(range(self.args.num_gpu)))
        model = model.cuda()

        print("Model = %s" % str(model))

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            # self.best_model_dict = torch.load(self.args.model_dir)['teacher']
            # self.best_model_dict = torch.load(self.args.model_dir)['model']
            
        else:
            print('random init params')
            if self.args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(model.state_dict())
        return model

    def calculate_model_complexity(self):
        args = self.args

        #judge the image size
        if args.dataset == "mini_imagenet":
            img_size = 224
        elif args.dataset == "cifar100":
            img_size = 224
        else:
            img_size = 224

        # calculate model complexity
        flops = FlopCountAnalysis(self.model, torch.rand(1, 3, img_size, img_size).cuda())
        print("FLOPs(million): ", flops.total()/1e6)
        print("parameters: ", parameter_count_table(self.model))

    def test(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)

            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label

                self.model.load_state_dict(self.best_model_dict)



                self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                self.best_model_dict = deepcopy(self.model.state_dict())
                self.model.module.mode = 'avg_cos'
                tsl, tsa = test(self.model, testloader, 0, args, session)
                if (tsa * 100) >= self.trlog['max_acc'][session]:
                    self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                    print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))


            else:  # incremental learning sessions
                print("testing session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)

                tsl, tsa = test(self.model, testloader, 0, args, session)

                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

            print(self.trlog['max_acc'])
            t_end_time = time.time()
            total_time = (t_end_time - t_start_time) / 60
            print('Total time used %.2f mins' % total_time)












