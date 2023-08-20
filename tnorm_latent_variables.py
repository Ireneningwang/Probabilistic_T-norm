import numpy as np
import torch
import torch.nn as nn
from tnorm_activation import latent_tnorm_module
from sklearn.model_selection import train_test_split
from tnorm_generate_data import generate
import logging 
import time

def accuracy(model, data_x, data_y, pct_close):
    n_items = len(data_y)
    X = torch.tensor(data_x)    # 2-D tensor   
    y = torch.tensor(data_y)    # actual as 1-D tensor
    output = model(X)           # all predicted as 2-D tensor
    pred = output.view(n_items) # all predicted as 1-D tensor
    n_correct = torch.sum(torch.abs(pred-y) - torch.abs(pct_close * y))
    result = (n_correct.item() * 100.0 / n_items)      # scalar
    return result


# define function to get gradient to norm
def get_gradient(model):
    total_norm = 0.
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm +=  param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm



def main(training_data, target_px, target_py1, target_py0):
    # Get statrted
    print("\n T norm using interpolation ratio method \n")
    torch.manual_seed(1)
    np.random.seed(1)

    # training_data = data.values
    train, test = train_test_split(training_data, test_size=0.2)
    logger.info(["Training data: ", train])
    logger.info(["Testing data: ", test])
    logger.info(["Training shape:", train.shape, "Testing shape: ",test.shape])

    # Create model
    print("\nCreating latent tnorm model \n")
    model = latent_tnorm_module(method='interpolation_ratio')

    # Train model
    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # in terms of the model parameters
   
    n_items = len(train)
    batches_per_epoch = n_items
    max_batches = 10 * batches_per_epoch
    print("Start training \n")
                
    # initialize two variables to store the outputs
    norms = torch.empty(max_batches)
    train_losses = torch.empty(max_batches)
    test_losses = torch.empty(max_batches)
    for epoch in range(max_batches):
        accum_loss = 0.0
        norm_sum = 0.
        train_dic = train.to_dict()
        for l_i, targets in train_dic.items():
            model.set_expression(str(l_i))
            for t_i in targets.values():
                # print(f"expr {l_i}, target {t_i}")
                # zero grad here for simplicity
                optimizer.zero_grad()
                # convert the target to tensor
                t_i = torch.tensor(t_i, dtype = torch.float, requires_grad=True)
                # make single prediction
                pred_i = model()
                # evaluate the loss
                loss_i = loss_func(pred_i, t_i)
                # print("\n", pred_i, t_i, loss_i)
                # accumulate the losses
                accum_loss = accum_loss + loss_i.item()
                # accumulate the gradient to norm
                norm_sum += get_gradient(model)
                # reset gradient then calculate gradients w.r.t. loss
                loss_i.backward()
                # update the parameters
                optimizer.step()

            else:
                # strore the training loss for each epoch
                train_losses[epoch] = accum_loss
                # strore average gradient to norm for each epoch
                norms[epoch] = norm_sum / len(train)

        # update ouput loss and/or for training data
        if epoch % (max_batches // 10) == 0:
            print("\n-------------------------------------------------------------------------")
            print("\nbatch = %6d" %epoch, end = "")
            print("\tTraining batch loss = %7.4f" % accum_loss, end="\n")
            logger.info("-------------------------------------------------------------------------")
            logger.info(["batch = ", epoch])
            logger.info(["Training batch loss = ", accum_loss])
            # get model parameters: px, py, dependency
            for name, parameter in model.named_parameters():
                print(name, ":", parameter)
                logger.info([name, ":", parameter])
            accum_loss = 0
        
        # Validation  
        with torch.no_grad():
            accum_test_loss = 0
            test_dic = test.to_dict()
            for l_i, targets in test_dic.items():
                model.set_expression(str(l_i))
                for t_i in targets.values():
                    # convert the target to tensor
                    t_i = torch.tensor(t_i, dtype = torch.float, requires_grad=True)
                    # make predictions
                    pred_i = model()
                    # evaluate the loss
                    loss_i = loss_func(pred_i, t_i)
                    accum_test_loss = accum_test_loss + loss_i.item()
                else:
                    test_losses[epoch] = loss_i

            # update ouput loss and/or for training data
            if epoch % (max_batches // 10) == 0:
                print("\n-------------------------------------------------------------------------")
                print("\nbatch = %6d" %epoch, end = "")
                print("\tValidation batch loss = %7.4f" % accum_test_loss, end="\n")
                logger.info("-------------------------------------------------------------------------")
                logger.info(["batch = ", epoch])
                logger.info(["Validation batch loss = ", accum_test_loss])
                # get model parameters: px, py, dependency
                for name, parameter in model.named_parameters():
                    print(name, ":", parameter)
                    logger.info([name, ":", parameter])
          
    print("Training loss:\n", train_losses, "Testing Loss:\n", test_losses, "\n norms: \n",norms )
    print("\nTraining complete. \n")
    logger.info(["Training loss:", train_losses])
    logger.info(["Testing Loss:", test_losses])
    logger.info(["norms:",norms])
    logger.info("Training complete.")

    # Evaluate model
    model = model.eval()
    # acc = accuracy(model, test_x, test_y, 0.15)
    # print("Accuracy on test data = %0.2f%%" %acc)



if __name__ == "__main__":
    # the general log settings
    log_format = "%(asctime)s - %(filename)s - %(levelname)s: \n %(message)s"
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = './log/' + rq + '.log'
    logging.basicConfig(filename=log_name,
                        filemode='a',
                        format=log_format,
                        level = logging.INFO)
    logger = logging.getLogger()
    (px, py1, py0), data= generate()
    logger.info(["px, py1, py0:", px, py1, py0])
    logger.info(["data:", data])
    main(data, px, py1, py0)
