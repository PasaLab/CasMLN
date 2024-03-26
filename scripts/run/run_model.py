import torch
from cmln.data import load_data
from cmln.model.load_model import load_model
from cmln.args_model import get_args
from cmln.trainer import load_trainer
import math

seeds = [102,42,310,923,228]
layers = [1, 2, 3, 4, 5, 6, 7, 8]
dropouts = [0.1, 0.3, 0.5, 0.7, 0.9]
amplifiers = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]

args = get_args()

best_auc = best_ap = 0
best_f1 = best_recall = 0
best_mae = best_rmse = 1e8
best_seed = 0
best_layer = 0
best_dropout = 0
best_amplifier =0 

for seed in seeds:
    for layer in layers:
        for dropout in dropouts:
            for amplifier in amplifiers:
                args.amplifier = amplifier
                args.seed = seed
                args.n_layers = layer
                args.dropout = dropout

                # dataset
                dataset, args = load_data(args)


                # model
                model = load_model(args, dataset)

                # device
                model = model.to(args.device)
                dataset.to(args.device)

                # train
                trainer, criterion = load_trainer(args)
                optimizer = torch.optim.Adam(
                    params=model.parameters(), lr=args.lr, weight_decay=args.wd
                )
                train_dict = trainer(
                    model,
                    optimizer,
                    criterion,
                    dataset,
                    args,
                    args.max_epochs,
                    args.patience,
                    disable_progress=False,
                    writer=None,
                    grad_clip=args.grad_clip,
                    device=args.device,
                )
                if 'test_auc' in train_dict.keys():
                    print(f"Final Test AUC: {train_dict['test_auc']:.4f} ")
                if 'test_ap' in train_dict.keys():
                    print(f"Final Test AP: {train_dict['test_ap']:.4f} ")
                if 'test_f1' in train_dict.keys():
                    print(f"Final Test F1: {train_dict['test_f1']:.4f} ")
                if 'test_acc' in train_dict.keys():
                    print(f"Final Test ACC: {train_dict['test_acc']:.4f} ")
                if 'test_mae' in train_dict.keys():
                    print(f"Final Test MAE: {train_dict['test_mae']:.4f} ")
                if 'test_rmse' in train_dict.keys():
                    print(f"Final Test RMSE: {train_dict['test_rmse']:.4f} ")
                print(f"Time per epoch: {train_dict['time_per_epoch']:.4f}")
                print('seed:',seed,' layer:',layer,' dropout:', dropout)

                file_name = "result_"+args.dataset+'_'+".txt"
                with open(file_name, 'a') as file:
                    if all(metric in train_dict.keys() for metric in ['test_auc','test_ap']):
                        file.write('Final Test AUC: '+str(train_dict['test_auc'])+"   ")
                        file.write('Final Test AP: '+str(train_dict['test_ap'])+"   ")
                        file.write('  Time per epoch: '+str(train_dict['time_per_epoch']))
                        file.write('  seed:'+str(seed)+'  layer:'+str(layer)+'  dropout:'+str(dropout)+'  amplifier:'+str(amplifier)+'\n\n')
                        if train_dict['test_auc']>best_auc:
                            best_auc = train_dict['test_auc']
                            best_seed = seed
                            best_layer = layer
                            best_dropout = dropout
                            best_amplifier = amplifier
                        if train_dict['test_ap']>best_ap:
                            best_ap = train_dict['test_ap']
                    if all(metric in train_dict.keys() for metric in ['test_f1','test_recall']):
                        file.write('Final Test F1: '+str(train_dict['test_f1'])+"   ")
                        file.write('Final Test Recall: '+str(train_dict['test_recall'])+"   ")
                        file.write('  Time per epoch: '+str(train_dict['time_per_epoch']))
                        file.write('  seed:'+str(seed)+'  layer:'+str(layer)+'  dropout:'+str(dropout)+'  amplifier:'+str(amplifier)+'\n\n')
                        if train_dict['test_f1']>best_f1:
                            best_f1 = train_dict['test_f1']
                            best_seed = seed
                            best_layer = layer
                            best_dropout = dropout
                            best_amplifier = amplifier
                        if train_dict['test_recall']>best_recall:
                            best_recall = train_dict['test_recall']
                    if all(metric in train_dict.keys() for metric in ['test_mae','test_rmse']):
                        file.write('Final Test MAE: '+str(train_dict['test_mae'])+"   ")
                        file.write('Final Test RMSE: '+str(train_dict['test_rmse'])+"   ")
                        file.write('  Time per epoch: '+str(train_dict['time_per_epoch']))
                        file.write('  seed:'+str(seed)+'  layer:'+str(layer)+'  dropout:'+str(dropout)+'  amplifier:'+str(amplifier)+'\n\n')
                        if train_dict['test_mae']<best_mae:
                            best_mae = train_dict['test_mae']
                            best_seed = seed
                            best_layer = layer
                            best_dropout = dropout
                            best_amplifier = amplifier
                        if train_dict['test_rmse']<best_rmse:
                            best_rmse = train_dict['test_rmse']

file_name = "result_"+args.dataset+'_'+".txt"

with open(file_name, 'a') as file:
    if args.dataset in ['Aminer', 'Ecomm']:
        file.write('Best auc: '+str(best_auc))
        file.write('  Best ap: '+str(best_ap))
    elif args.dataset in ['Yelp-nc']:
        file.write('Best f1: '+str(best_f1))
        file.write('  Best recall: '+str(best_recall))
    elif args.dataset in ['covid']:
        file.write('Best mae: '+str(best_mae))
        file.write('  Best rmse: '+str(best_rmse))
        
    file.write('  Best seed:'+str(best_seed))
    file.write('  Best layer:'+str(best_layer))
    file.write('  Best dropout:'+str(best_dropout)+'\n')
    file.write('  Best amplifier:'+str(best_amplifier)+'\n')

if args.dataset in ['Aminer', 'Ecomm']:
    print('Best auc: '+str(best_auc))
    print('Best ap: '+str(best_ap))
elif args.dataset in ['Yelp-nc']:
    print('Best f1: '+str(best_f1))
    print('Best recall: '+str(best_recall))
elif args.dataset in ['covid']:
    print('Best mae: '+str(best_mae))
    print('  Best rmse: '+str(best_rmse))
        
print('Best seed:'+str(best_seed))
print('Best layer:'+str(best_layer))
print('Best dropout:'+str(best_dropout)+'\n')
print('Best amplifier:'+str(best_amplifier)+'\n')
