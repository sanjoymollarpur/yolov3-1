#first page
import matplotlib.pyplot as plt
from time import sleep
import config
import torch
import torch.optim as optim


from model import yv3
from tqdm import tqdm
from utils import (
    intersection_over_union,
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)

from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    losses1=0
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update progress bar
        mean_loss = sum(losses) / len(losses)
        losses1=mean_loss
        loop.set_postfix(loss=mean_loss)
    return losses1

def test_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
    loop = tqdm(train_loader, leave=True)
    losses = []
    losses1=0
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y0, y1, y2 = (
            y[0].to(config.DEVICE),
            y[1].to(config.DEVICE),
            y[2].to(config.DEVICE),
        )             

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = (
                loss_fn(out[0], y0, scaled_anchors[0])
                + loss_fn(out[1], y1, scaled_anchors[1])
                + loss_fn(out[2], y2, scaled_anchors[2])
            )

        losses.append(loss.item())
        
        # update progress bar
        mean_loss = sum(losses) / len(losses)
        losses1=mean_loss
        loop.set_postfix(loss=mean_loss)
    return losses1


def main():
    # model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    # lr1=[]
    # lr_iou=[]
    # lr_f1=[]
    # lr_ap=[]
     
    for thres in config.CONF_THRESHOLD:

        lr1=[]
        lr_iou=[]
        lr_f1=[]
        lr_ap=[]
        lr_max_ap=[]
        import time
        start=time.time()
        for lrate in config.LEARNING_RATE:
            
            model = yv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
            lr1.append(lrate)
            optimizer = optim.Adam(
                model.parameters(), lr=lrate, weight_decay=config.WEIGHT_DECAY
            )
            loss_fn = YoloLoss()
            scaler = torch.cuda.amp.GradScaler()

            train_loader, test_loader, train_eval_loader = get_loaders(
                train_csv_path=config.DATASET + "/neg-aug-train.csv", test_csv_path=config.DATASET + "/neg-aug-test.csv" #train-aug -> train, test-aug -> test
            )

            if config.LOAD_MODEL:
                load_checkpoint(
                    config.CHECKPOINT_FILE, model, optimizer, lrate
                )
                
            scaled_anchors = (
                torch.tensor(config.ANCHORS)
                * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
            ).to(config.DEVICE)

            #for epoch in range(config.NUM_EPOCHS):
            acc =[]
            val_acc=[]
            loss=[]
            val_loss=[]
            no_obj=[]
            val_no_obj=[]
            map1=[]
            val_map1=[]
            iou1=[]
            val_iou1=[]
            f1=[]
            val_f1=[]
            ap_max=0
            
            for epoch in range(config.NUM_EPOCHS):
            
                stat1=time.time() 
                if config.LOAD_MODEL:
                    plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
                    import sys
                    sys.exit()
                    
                print(f"Currently epoch {epoch}/{config.NUM_EPOCHS} Threshold {thres}, lr {lrate}")
                loss1=train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
                
                loss.append(loss1)
                pred_boxes, true_boxes = get_evaluation_bboxes(
                    train_loader,
                    model,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=thres,
                )
                
                
                print("hello-train")
                mapval, iou11, f11 = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                    num_classes=config.NUM_CLASSES,
                )

                print(f"Train AP: {mapval.item()}, IOU: {iou11}, f1: {f11}")
                f1.append(f11)
                iou1.append(iou11)
                map1.append(mapval)
                model.eval()
                loss2=test_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
                val_loss.append(loss2)
                print("test", iou11)
                pred_boxes, true_boxes = get_evaluation_bboxes(
                    test_loader,
                    model,
                    iou_threshold=config.NMS_IOU_THRESH,
                    anchors=config.ANCHORS,
                    threshold=thres,
                )
                
                
                print("hello-train")
                mapval, iou11, f11 = mean_average_precision(
                    pred_boxes,
                    true_boxes,
                    iou_threshold=config.MAP_IOU_THRESH,
                    box_format="midpoint",
                    num_classes=config.NUM_CLASSES,
                )

                print("test", iou11)
                val_f1.append(f11)
                val_iou1.append(iou11)
                val_map1.append(mapval)
                print(f"Test AP: {mapval.item()}, IOU: {iou11}, f1: {f11}")
                if mapval.item()>ap_max:
                    ap_max=mapval.item()

                if config.SAVE_MODEL and mapval.item() >= ap_max:
                    save_checkpoint(model, optimizer, filename=f"weight/wt_max200_lr-{lrate}_threshold-{thres}.pth.tar")
                    import time
                    sleep(1)

                if config.SAVE_MODEL and mapval.item() >= 0.7:
                    save_checkpoint(model, optimizer, filename=f"weight/wt200_lr-{lrate}_threshold-{thres}.pth.tar")
                    import time
                    sleep(1)

                model.train()
            lr_max_ap.append(ap_max)
            lr_iou.append(val_iou1[config.NUM_EPOCHS-1])
            lr_f1.append(val_f1[config.NUM_EPOCHS-1])
            lr_ap.append(val_map1[config.NUM_EPOCHS-1])
            

            class_acc = iou1
            val_class_acc = val_iou1
            epochs = range(1, config.NUM_EPOCHS+1)
            plt.plot(epochs, class_acc, 'y', label='Training iou')
            plt.plot(epochs, val_class_acc, 'r', label='Validation iou')
            plt.grid()
            plt.title(f'Training and validation iou lr-{lrate} threshold-{thres}')
            plt.xlabel('Epochs')
            plt.ylabel('iou')
            plt.legend()
            plt.savefig(f"graph2/iou_lr-{lrate}_threshold-{thres}.png")
            plt.close()

            class_acc = f1
            val_class_acc = val_f1
            epochs = range(1, config.NUM_EPOCHS+1)
            plt.plot(epochs, class_acc, 'y', label='Training dsc')
            plt.plot(epochs, val_class_acc, 'r', label='Validation dsc')
            class_acc = iou1
            val_class_acc = val_iou1
            plt.plot(epochs, class_acc, 'orange', label='Training iou')
            plt.plot(epochs, val_class_acc, 'pink', label='Validation iou')
            plt.grid()
            plt.title(f'Training and validation dsc and iou lr-{lrate} threshold-{thres}')
            plt.xlabel('Epochs')
            plt.ylabel('dsc, iou')
            plt.legend()
            plt.savefig(f"graph2/dsc_iou_lr-{lrate}_threshold-{thres}.png")
            plt.close()

            class_acc = f1
            val_class_acc = val_f1
            epochs = range(1, config.NUM_EPOCHS+1)
            plt.plot(epochs, class_acc, 'y', label='Training dsc')
            plt.plot(epochs, val_class_acc, 'r', label='Validation dsc')
            plt.grid()
            plt.title(f'Training and validation dsc lr-{lrate} threshold-{thres}')
            plt.xlabel('Epochs')
            plt.ylabel('dsc')
            plt.legend()
            plt.savefig(f"graph2/dsc_lr-{lrate}_threshold-{thres}.png")
            plt.close()


            class_acc = map1
            val_class_acc = val_map1
            epochs = range(1, config.NUM_EPOCHS+1)
            plt.plot(epochs, class_acc, 'y', label='Training AP')
            plt.plot(epochs, val_class_acc, 'r', label='Validation AP')
            plt.grid()
            plt.title(f'Training and validation AP lr-{lrate} threshold-{thres}')
            plt.xlabel('Epochs')
            plt.ylabel('AP')
            plt.legend()
            plt.savefig(f"graph2/ap_lr-{lrate}_threshold-{thres}.png")
            plt.close()


            class_acc = loss
            val_class_acc = val_loss
            epochs = range(1, config.NUM_EPOCHS+1)
            plt.plot(epochs, class_acc, 'y', label='Training loss')
            plt.plot(epochs, val_class_acc, 'r', label='Validation  loss')
            plt.grid()
            plt.title(f'Training and validation loss lr-{lrate} threshold-{thres}')
            plt.xlabel('Epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(f"graph2/loss_map_lr-{lrate}_threshold-{thres}.png")
            plt.close()
            end1 = time.time()
            print("Time for one epoch is: ", end1-stat1)
        ap_max_l=lr_max_ap
        class_acc = lr_ap
        val_class_acc = lr_f1
        val_class_iou = lr_iou
        lr_ap_val=[]
        lr_max_ap_val=[]
        for data in lr_max_ap:
            lr_max_ap_val.append(data)
        
        for data in lr_ap:
            lr_ap_val.append(data)
        lr_f1_val=[]
        for data in lr_f1:
            lr_f1_val.append(data)
        lr_iou_val=[]
        for data in lr_iou:
            lr_iou_val.append(data)
        # epochs = range(1, config.NUM_EPOCHS+1)
        plt.plot(lr1, ap_max_l, 'g', label='ap_max')
        plt.plot(lr1, class_acc, 'y', label='ap')
        plt.plot(lr1, val_class_acc, 'r', label='f1')
        plt.plot(lr1, val_class_iou, 'b', label='iou')
        plt.grid()
        plt.title(f"metric vs lr threshold-{thres}")
        plt.xlabel('lr')
        plt.ylabel('ap, iou, f1')
        plt.legend()
        plt.savefig(f"graph2/ap_iou_f1_threshold-{thres}.png")
        plt.close()
        with open(f"metric/metric_threshold-{thres}.txt", 'w') as file:
            file.writelines(f"Threshold: {thres}")
            file.writelines("\n")
            file.writelines("Learning Rate: "+ str(config.LEARNING_RATE))
            file.writelines("\n")
            file.writelines("AP           : "+ str(lr_ap_val))
            file.writelines("\n")
            file.writelines("AP max       : "+ str(lr_max_ap_val))
            file.writelines("\n")
            file.writelines("DSC          : "+ str(lr_f1_val))
            file.writelines("\n")
            file.writelines("IOU          : "+ str(lr_iou_val))
            file.close()
        end = time.time()
        print("Time for one learning rate epoch is: ", end-start)



if __name__ == "__main__":
    main()

















# import matplotlib.pyplot as plt
# from time import sleep
# import config
# import torch
# import torch.optim as optim

# from model import yv3
# from tqdm import tqdm
# from utils import (
#     intersection_over_union,
#     mean_average_precision,
#     cells_to_bboxes,
#     get_evaluation_bboxes,
#     save_checkpoint,
#     load_checkpoint,
#     check_class_accuracy,
#     get_loaders,
#     plot_couple_examples
# )
# from loss import YoloLoss
# import warnings
# warnings.filterwarnings("ignore")

# torch.backends.cudnn.benchmark = True


# def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
#     loop = tqdm(train_loader, leave=True)
#     losses = []
#     losses1=0
#     for batch_idx, (x, y) in enumerate(loop):
#         x = x.to(config.DEVICE)
#         y0, y1, y2 = (
#             y[0].to(config.DEVICE),
#             y[1].to(config.DEVICE),
#             y[2].to(config.DEVICE),
#         )

#         with torch.cuda.amp.autocast():
#             out = model(x)
#             loss = (
#                 loss_fn(out[0], y0, scaled_anchors[0])
#                 + loss_fn(out[1], y1, scaled_anchors[1])
#                 + loss_fn(out[2], y2, scaled_anchors[2])
#             )

#         losses.append(loss.item())
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # update progress bar
#         mean_loss = sum(losses) / len(losses)
#         losses1=mean_loss
#         loop.set_postfix(loss=mean_loss)
#     return losses1

# def test_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
#     loop = tqdm(train_loader, leave=True)
#     losses = []
#     losses1=0
#     for batch_idx, (x, y) in enumerate(loop):
#         x = x.to(config.DEVICE)
#         y0, y1, y2 = (
#             y[0].to(config.DEVICE),
#             y[1].to(config.DEVICE),
#             y[2].to(config.DEVICE),
#         )

#         with torch.cuda.amp.autocast():
#             out = model(x)
#             loss = (
#                 loss_fn(out[0], y0, scaled_anchors[0])
#                 + loss_fn(out[1], y1, scaled_anchors[1])
#                 + loss_fn(out[2], y2, scaled_anchors[2])
#             )

#         losses.append(loss.item())
        
#         # update progress bar
#         mean_loss = sum(losses) / len(losses)
#         losses1=mean_loss
#         loop.set_postfix(loss=mean_loss)
#     return losses1


# def main():
#     # model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
#     # lr1=[]
#     # lr_iou=[]
#     # lr_f1=[]
#     # lr_ap=[]
#     for thres in config.CONF_THRESHOLD:

#         lr1=[]
#         lr_iou=[]
#         lr_f1=[]
#         lr_ap=[]
#         lr_max_ap=[]
#         for lrate in config.LEARNING_RATE:
#             model = yv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
#             lr1.append(lrate)
#             optimizer = optim.Adam(
#                 model.parameters(), lr=lrate, weight_decay=config.WEIGHT_DECAY
#             )
#             loss_fn = YoloLoss()
#             scaler = torch.cuda.amp.GradScaler()

#             train_loader, test_loader, train_eval_loader = get_loaders(
#                 train_csv_path=config.DATASET + "/pos-neg-train.csv", test_csv_path=config.DATASET + "/pos-neg-test.csv" #train-aug -> train, test-aug -> test
#             )

#             if config.LOAD_MODEL:
#                 load_checkpoint(
#                     config.CHECKPOINT_FILE, model, optimizer, lrate
#                 )
                
#             scaled_anchors = (
#                 torch.tensor(config.ANCHORS)
#                 * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#             ).to(config.DEVICE)

#             #for epoch in range(config.NUM_EPOCHS):
#             acc =[]
#             val_acc=[]
#             loss=[]
#             val_loss=[]
#             no_obj=[]
#             val_no_obj=[]
#             map1=[]
#             val_map1=[]
#             iou1=[]
#             val_iou1=[]
#             f1=[]
#             val_f1=[]
#             ap_max=0
#             for epoch in range(config.NUM_EPOCHS):
#             #for epoch in range(2):
#                 if config.LOAD_MODEL:
#                     plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
                    
#                     import sys
#                     sys.exit()
                    
#                 print(f"Currently epoch {epoch}/{config.NUM_EPOCHS} Threshold {thres}, lr {lrate}")
#                 loss1=train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
                
#                 loss.append(loss1)
#                 # pred_boxes, true_boxes = get_evaluation_bboxes(
#                 #     train_loader,
#                 #     model,
#                 #     iou_threshold=config.NMS_IOU_THRESH,
#                 #     anchors=config.ANCHORS,
#                 #     threshold=thres,
#                 # )
                 
                
#                 # print("hello-train")
#                 # mapval, iou11, f11 = mean_average_precision(
#                 #     pred_boxes,
#                 #     true_boxes,
#                 #     iou_threshold=config.MAP_IOU_THRESH,
#                 #     box_format="midpoint",
#                 #     num_classes=config.NUM_CLASSES,
#                 # )
#                 # print(f"Train AP: {mapval.item()}, IOU: {iou11}, f1: {f11}")
#                 # f1.append(f11)
#                 # iou1.append(iou11)
#                 # map1.append(mapval)
#                 # model.eval()
#                 # # loss2=test_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
#                 # val_loss.append(loss2)
#                 # print("test", iou11)
#                 pred_boxes, true_boxes = get_evaluation_bboxes(
#                     test_loader,
#                     model,
#                     iou_threshold=config.NMS_IOU_THRESH,
#                     anchors=config.ANCHORS,
#                     threshold=thres,
#                 )
                
                
#                 print("hello-train")
#                 mapval, iou11, f11 = mean_average_precision(
#                     pred_boxes,
#                     true_boxes,
#                     iou_threshold=config.MAP_IOU_THRESH,
#                     box_format="midpoint",
#                     num_classes=config.NUM_CLASSES,
#                 )
#                 # print("test", iou11)
#                 # val_f1.append(f11)
#                 # val_iou1.append(iou11)
#                 val_map1.append(mapval)
#                 print(f"Test AP: {mapval.item()}")
#                 if mapval.item()>ap_max:
#                     ap_max=mapval.item()

#                 if config.SAVE_MODEL and mapval.item() >= ap_max:
#                     save_checkpoint(model, optimizer, filename=f"weight/wt_max2_lr-{lrate}_threshold-{thres}.pth.tar")
#                     import time
#                     sleep(1)

#                 if config.SAVE_MODEL and mapval.item() >= 0.7:
#                     save_checkpoint(model, optimizer, filename=f"weight/wt2_lr-{lrate}_threshold-{thres}.pth.tar")
#                     import time
#                     sleep(1)

#                 model.train()
#             lr_max_ap.append(ap_max)
#             # lr_iou.append(val_iou1[config.NUM_EPOCHS-1])
#             # lr_f1.append(val_f1[config.NUM_EPOCHS-1])
#             lr_ap.append(val_map1[config.NUM_EPOCHS-1])
            
            
#             # class_acc = iou1
#             # val_class_acc = val_iou1
#             # epochs = range(1, config.NUM_EPOCHS+1)
#             # plt.plot(epochs, class_acc, 'y', label='Training iou')
#             # plt.plot(epochs, val_class_acc, 'r', label='Validation iou')
#             # plt.grid()
#             # plt.title(f'Training and validation iou lr-{lrate} threshold-{thres}')
#             # plt.xlabel('Epochs')
#             # plt.ylabel('iou')
#             # plt.legend()
#             # plt.savefig(f"graph2/iou_lr-{lrate}_threshold-{thres}.png")
#             # plt.close()

#             # class_acc = f1
#             # val_class_acc = val_f1
#             # epochs = range(1, config.NUM_EPOCHS+1)
#             # plt.plot(epochs, class_acc, 'y', label='Training dsc')
#             # plt.plot(epochs, val_class_acc, 'r', label='Validation dsc')
#             # class_acc = iou1
#             # val_class_acc = val_iou1
#             # plt.plot(epochs, class_acc, 'orange', label='Training iou')
#             # plt.plot(epochs, val_class_acc, 'pink', label='Validation iou')

#             # plt.grid()
#             # plt.title(f'Training and validation dsc and iou lr-{lrate} threshold-{thres}')
#             # plt.xlabel('Epochs')
#             # plt.ylabel('dsc, iou')
#             # plt.legend()
#             # plt.savefig(f"graph2/dsc_iou_lr-{lrate}_threshold-{thres}.png")
#             # plt.close()

#             # class_acc = f1
#             # val_class_acc = val_f1
#             # epochs = range(1, config.NUM_EPOCHS+1)
#             # plt.plot(epochs, class_acc, 'y', label='Training dsc')
#             # plt.plot(epochs, val_class_acc, 'r', label='Validation dsc')
#             # plt.grid()
#             # plt.title(f'Training and validation dsc lr-{lrate} threshold-{thres}')
#             # plt.xlabel('Epochs')
#             # plt.ylabel('dsc')
#             # plt.legend()
#             # plt.savefig(f"graph2/dsc_lr-{lrate}_threshold-{thres}.png")
#             # plt.close()


#             #class_acc = map1
#             val_class_acc = val_map1
#             epochs = range(1, config.NUM_EPOCHS+1)
#             #plt.plot(epochs, class_acc, 'y', label='Training AP')
#             plt.plot(epochs, val_class_acc, 'r', label='Validation AP')
#             plt.grid()
#             plt.title(f'Training and validation AP lr-{lrate} threshold-{thres}')
#             plt.xlabel('Epochs')
#             plt.ylabel('AP')
#             plt.legend()
#             plt.savefig(f"graph2/ap_lr-{lrate}_threshold-{thres}.png")
#             plt.close()


#             # class_acc = loss
#             # val_class_acc = val_loss
#             # epochs = range(1, config.NUM_EPOCHS+1)
#             # plt.plot(epochs, class_acc, 'y', label='Training loss')
#             # plt.plot(epochs, val_class_acc, 'r', label='Validation  loss')
#             # plt.grid()
#             # plt.title(f'Training and validation loss lr-{lrate} threshold-{thres}')
#             # plt.xlabel('Epochs')
#             # plt.ylabel('loss')
#             # plt.legend()
#             # plt.savefig(f"graph2/loss_map_lr-{lrate}_threshold-{thres}.png")
#             # plt.close()
#         ap_max_l=lr_max_ap
#         class_acc = lr_ap
#         # val_class_acc = lr_f1
#         # val_class_iou = lr_iou
#         lr_ap_val=[]
#         lr_max_ap_val=[]
#         for data in lr_max_ap:
#             lr_max_ap_val.append(data)
        
#         for data in lr_ap:
#             lr_ap_val.append(data)
#         # lr_f1_val=[]
#         # for data in lr_f1:
#         #     lr_f1_val.append(data)
#         # lr_iou_val=[]
#         # for data in lr_iou:
#         #     lr_iou_val.append(data)
#         epochs = range(1, config.NUM_EPOCHS+1)
#         plt.plot(lr1, ap_max_l, 'g', label='ap_max')
#         plt.plot(lr1, class_acc, 'y', label='ap')
#         # plt.plot(lr1, val_class_acc, 'r', label='f1')
#         # plt.plot(lr1, val_class_iou, 'b', label='iou')
#         plt.grid()
#         plt.title(f"metric vs lr threshold-{thres}")
#         plt.xlabel('lr')
#         plt.ylabel('ap, iou, f1')
#         plt.legend()
#         plt.savefig(f"graph2/ap_iou_f1_threshold-{thres}.png")
#         plt.close()
#         with open(f"metric/metric_threshold-{thres}.txt", 'w') as file:
#             file.writelines(f"Threshold: {thres}")
#             file.writelines("\n")
#             file.writelines("Learning Rate: "+ str(config.LEARNING_RATE))
#             file.writelines("\n")
#             file.writelines("AP           : "+ str(lr_ap_val))
#             file.writelines("\n")
#             file.writelines("AP max       : "+ str(lr_max_ap_val))
#             # file.writelines("\n")
#             # file.writelines("DSC          : "+ str(lr_f1_val))
#             # file.writelines("\n")
#             # file.writelines("IOU          : "+ str(lr_iou_val))
#             file.close()



# if __name__ == "__main__":
#     main()































# import matplotlib.pyplot as plt
# from time import sleep
# import config
# import torch
# import torch.optim as optim

# from model import yv3
# from tqdm import tqdm
# from utils import (
#     intersection_over_union,
#     mean_average_precision,
#     cells_to_bboxes,
#     get_evaluation_bboxes,
#     save_checkpoint,
#     load_checkpoint,
#     check_class_accuracy,
#     get_loaders,
#     plot_couple_examples
# )
# from loss import YoloLoss
# import warnings
# warnings.filterwarnings("ignore")

# torch.backends.cudnn.benchmark = True


# def train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
#     loop = tqdm(train_loader, leave=True)
#     losses = []
#     losses1=0
#     for batch_idx, (x, y) in enumerate(loop):
#         x = x.to(config.DEVICE)
#         y0, y1, y2 = (
#             y[0].to(config.DEVICE),
#             y[1].to(config.DEVICE),
#             y[2].to(config.DEVICE),
#         )

#         with torch.cuda.amp.autocast():
#             out = model(x)
#             loss = (
#                 loss_fn(out[0], y0, scaled_anchors[0])
#                 + loss_fn(out[1], y1, scaled_anchors[1])
#                 + loss_fn(out[2], y2, scaled_anchors[2])
#             )

#         losses.append(loss.item())
#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()

#         # update progress bar
#         mean_loss = sum(losses) / len(losses)
#         losses1=mean_loss
#         loop.set_postfix(loss=mean_loss)
#     return losses1

# def test_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors):
#     loop = tqdm(train_loader, leave=True)
#     losses = []
#     losses1=0
#     for batch_idx, (x, y) in enumerate(loop):
#         x = x.to(config.DEVICE)
#         y0, y1, y2 = (
#             y[0].to(config.DEVICE),
#             y[1].to(config.DEVICE),
#             y[2].to(config.DEVICE),
#         )

#         with torch.cuda.amp.autocast():
#             out = model(x)
#             loss = (
#                 loss_fn(out[0], y0, scaled_anchors[0])
#                 + loss_fn(out[1], y1, scaled_anchors[1])
#                 + loss_fn(out[2], y2, scaled_anchors[2])
#             )

#         losses.append(loss.item())
        
#         # update progress bar
#         mean_loss = sum(losses) / len(losses)
#         losses1=mean_loss
#         loop.set_postfix(loss=mean_loss)
#     return losses1


# def main():
#     # model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
#     # lr1=[]
#     # lr_iou=[]
#     # lr_f1=[]
#     # lr_ap=[]
#     for thres in config.CONF_THRESHOLD:

#         lr1=[]
#         lr_iou=[]
#         lr_f1=[]
#         lr_ap=[]
#         lr_max_ap=[]
#         for lrate in config.LEARNING_RATE:
#             model = yv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
#             lr1.append(lrate)
#             optimizer = optim.Adam(
#                 model.parameters(), lr=lrate, weight_decay=config.WEIGHT_DECAY
#             )
#             loss_fn = YoloLoss()
#             scaler = torch.cuda.amp.GradScaler()

#             train_loader, test_loader, train_eval_loader = get_loaders(
#                 train_csv_path=config.DATASET + "/train-aug.csv", test_csv_path=config.DATASET + "/test-aug.csv" #train-aug -> train, test-aug -> test
#             )

#             if config.LOAD_MODEL:
#                 load_checkpoint(
#                     config.CHECKPOINT_FILE, model, optimizer, lrate
#                 )
                
#             scaled_anchors = (
#                 torch.tensor(config.ANCHORS)
#                 * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
#             ).to(config.DEVICE)

#             #for epoch in range(config.NUM_EPOCHS):
#             acc =[]
#             val_acc=[]
#             loss=[]
#             val_loss=[]
#             no_obj=[]
#             val_no_obj=[]
#             map1=[]
#             val_map1=[]
#             iou1=[]
#             val_iou1=[]
#             f1=[]
#             val_f1=[]
#             ap_max=0
#             for epoch in range(config.NUM_EPOCHS):
#             #for epoch in range(2):
#                 if config.LOAD_MODEL:
#                     plot_couple_examples(model, test_loader, 0.6, 0.5, scaled_anchors)
                    
#                     import sys
#                     sys.exit()
                    
#                 print(f"Currently epoch {epoch}/{config.NUM_EPOCHS} Threshold {thres}, lr {lrate}")
#                 loss1=train_fn(train_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
                
#                 loss.append(loss1)
#                 pred_boxes, true_boxes = get_evaluation_bboxes(
#                     train_loader,
#                     model,
#                     iou_threshold=config.NMS_IOU_THRESH,
#                     anchors=config.ANCHORS,
#                     threshold=thres,
#                 )
                
                
#                 print("hello-train")
#                 mapval, iou11, f11 = mean_average_precision(
#                     pred_boxes,
#                     true_boxes,
#                     iou_threshold=config.MAP_IOU_THRESH,
#                     box_format="midpoint",
#                     num_classes=config.NUM_CLASSES,
#                 )
#                 print(f"Train AP: {mapval.item()}, IOU: {iou11}, f1: {f11}")
#                 f1.append(f11)
#                 iou1.append(iou11)
#                 map1.append(mapval)
#                 model.eval()
#                 loss2=test_fn(test_loader, model, optimizer, loss_fn, scaler, scaled_anchors)
#                 val_loss.append(loss2)
#                 print("test", iou11)
#                 pred_boxes, true_boxes = get_evaluation_bboxes(
#                     test_loader,
#                     model,
#                     iou_threshold=config.NMS_IOU_THRESH,
#                     anchors=config.ANCHORS,
#                     threshold=thres,
#                 )
                
                
#                 print("hello-train")
#                 mapval, iou11, f11 = mean_average_precision(
#                     pred_boxes,
#                     true_boxes,
#                     iou_threshold=config.MAP_IOU_THRESH,
#                     box_format="midpoint",
#                     num_classes=config.NUM_CLASSES,
#                 )
#                 print("test", iou11)
#                 val_f1.append(f11)
#                 val_iou1.append(iou11)
#                 val_map1.append(mapval)
#                 print(f"Test AP: {mapval.item()}, IOU: {iou11}, f1: {f11}")
#                 if mapval.item()>ap_max:
#                     ap_max=mapval.item()

#                 if config.SAVE_MODEL and mapval.item() >= ap_max:
#                     save_checkpoint(model, optimizer, filename=f"weight/wt_max200_lr-{lrate}_threshold-{thres}.pth.tar")
#                     import time
#                     sleep(1)

#                 if config.SAVE_MODEL and mapval.item() >= 0.7:
#                     save_checkpoint(model, optimizer, filename=f"weight/wt200_lr-{lrate}_threshold-{thres}.pth.tar")
#                     import time
#                     sleep(1)

#                 model.train()
#             lr_max_ap.append(ap_max)
#             lr_iou.append(val_iou1[config.NUM_EPOCHS-1])
#             lr_f1.append(val_f1[config.NUM_EPOCHS-1])
#             lr_ap.append(val_map1[config.NUM_EPOCHS-1])
            
            
#             class_acc = iou1
#             val_class_acc = val_iou1
#             epochs = range(1, config.NUM_EPOCHS+1)
#             plt.plot(epochs, class_acc, 'y', label='Training iou')
#             plt.plot(epochs, val_class_acc, 'r', label='Validation iou')
#             plt.grid()
#             plt.title(f'Training and validation iou lr-{lrate} threshold-{thres}')
#             plt.xlabel('Epochs')
#             plt.ylabel('iou')
#             plt.legend()
#             plt.savefig(f"graph2/iou_lr-{lrate}_threshold-{thres}.png")
#             plt.close()

#             class_acc = f1
#             val_class_acc = val_f1
#             epochs = range(1, config.NUM_EPOCHS+1)
#             plt.plot(epochs, class_acc, 'y', label='Training dsc')
#             plt.plot(epochs, val_class_acc, 'r', label='Validation dsc')
#             class_acc = iou1
#             val_class_acc = val_iou1
#             plt.plot(epochs, class_acc, 'orange', label='Training iou')
#             plt.plot(epochs, val_class_acc, 'pink', label='Validation iou')

#             plt.grid()
#             plt.title(f'Training and validation dsc and iou lr-{lrate} threshold-{thres}')
#             plt.xlabel('Epochs')
#             plt.ylabel('dsc, iou')
#             plt.legend()
#             plt.savefig(f"graph2/dsc_iou_lr-{lrate}_threshold-{thres}.png")
#             plt.close()

#             class_acc = f1
#             val_class_acc = val_f1
#             epochs = range(1, config.NUM_EPOCHS+1)
#             plt.plot(epochs, class_acc, 'y', label='Training dsc')
#             plt.plot(epochs, val_class_acc, 'r', label='Validation dsc')
#             plt.grid()
#             plt.title(f'Training and validation dsc lr-{lrate} threshold-{thres}')
#             plt.xlabel('Epochs')
#             plt.ylabel('dsc')
#             plt.legend()
#             plt.savefig(f"graph2/dsc_lr-{lrate}_threshold-{thres}.png")
#             plt.close()


#             class_acc = map1
#             val_class_acc = val_map1
#             epochs = range(1, config.NUM_EPOCHS+1)
#             plt.plot(epochs, class_acc, 'y', label='Training AP')
#             plt.plot(epochs, val_class_acc, 'r', label='Validation AP')
#             plt.grid()
#             plt.title(f'Training and validation AP lr-{lrate} threshold-{thres}')
#             plt.xlabel('Epochs')
#             plt.ylabel('AP')
#             plt.legend()
#             plt.savefig(f"graph2/ap_lr-{lrate}_threshold-{thres}.png")
#             plt.close()


#             class_acc = loss
#             val_class_acc = val_loss
#             epochs = range(1, config.NUM_EPOCHS+1)
#             plt.plot(epochs, class_acc, 'y', label='Training loss')
#             plt.plot(epochs, val_class_acc, 'r', label='Validation  loss')
#             plt.grid()
#             plt.title(f'Training and validation loss lr-{lrate} threshold-{thres}')
#             plt.xlabel('Epochs')
#             plt.ylabel('loss')
#             plt.legend()
#             plt.savefig(f"graph2/loss_map_lr-{lrate}_threshold-{thres}.png")
#             plt.close()
#         ap_max_l=lr_max_ap
#         class_acc = lr_ap
#         val_class_acc = lr_f1
#         val_class_iou = lr_iou
#         lr_ap_val=[]
#         lr_max_ap_val=[]
#         for data in lr_max_ap:
#             lr_max_ap_val.append(data)
        
#         for data in lr_ap:
#             lr_ap_val.append(data)
#         lr_f1_val=[]
#         for data in lr_f1:
#             lr_f1_val.append(data)
#         lr_iou_val=[]
#         for data in lr_iou:
#             lr_iou_val.append(data)
#         #epochs = range(1, config.NUM_EPOCHS+1)
#         plt.plot(lr1, ap_max_l, 'g', label='ap_max')
#         plt.plot(lr1, class_acc, 'y', label='ap')
#         plt.plot(lr1, val_class_acc, 'r', label='f1')
#         plt.plot(lr1, val_class_iou, 'b', label='iou')
#         plt.grid()
#         plt.title(f"metric vs lr threshold-{thres}")
#         plt.xlabel('lr')
#         plt.ylabel('ap, iou, f1')
#         plt.legend()
#         plt.savefig(f"graph2/ap_iou_f1_threshold-{thres}.png")
#         plt.close()
#         with open(f"metric/metric_threshold-{thres}.txt", 'w') as file:
#             file.writelines(f"Threshold: {thres}")
#             file.writelines("\n")
#             file.writelines("Learning Rate: "+ str(config.LEARNING_RATE))
#             file.writelines("\n")
#             file.writelines("AP           : "+ str(lr_ap_val))
#             file.writelines("\n")
#             file.writelines("AP max       : "+ str(lr_max_ap_val))
#             file.writelines("\n")
#             file.writelines("DSC          : "+ str(lr_f1_val))
#             file.writelines("\n")
#             file.writelines("IOU          : "+ str(lr_iou_val))
#             file.close()



# if __name__ == "__main__":
#     main()


