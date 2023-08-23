import argparse
import utils.Audit as Audit
import utils.Purification as Forget
import sys

def parser():
    """

    :return: args
    """

    parser = argparse.ArgumentParser(prog='AFS')
    subparsers = parser.add_subparsers(help='sub-command help')

    parser_audit = subparsers.add_parser('audit')
    parser_audit.add_argument('--root',
                        default='../template/MNIST',
                        help='root dir to the project')
    parser_audit.add_argument('--query_label',
                        default='EXP1',
                        help='label of the query data, defined in Dataset.py/Config')
    parser_audit.add_argument('--cal_label',
                        default='CAL1',
                        help='label of the calibration data, defined in Dataset.py/Config')
    parser_audit.add_argument('--cal_test_label',
                        default='CALTEST1',
                        help='label of the calibration test data, defined in Dataset.py/Config')
    parser_audit.add_argument('--test_label',
                        default='TEST1',
                        help='label of the test data, defined in Dataset.py/Config')
    parser_audit.add_argument('--model2audit',
                        default='./models/base/best_model.pth',
                        help='relative path of model to be auditted to the root')
    parser_audit.add_argument('--model2cal',
                        default='./models/cal/best_model.pth',
                        help='relative path of the calibration model to the root')
    parser_audit.add_argument('--device',
                        default='cuda:0')
    parser_audit.add_argument('--KP_infer_batch_size',
                        type=int,
                        default=1024,
                        help='batch size for inference during membership attack')
    parser_audit.add_argument('--nclass',
                        type=int,
                        default=10,
                        help='number of classes')
    parser_audit.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='number of num_workers')
    parser_audit.add_argument('--command_class',
                              default=0,
                              type=int,
                              help='for internal use only, no change')

    parser_forget = subparsers.add_parser('forget')
    parser_forget.add_argument('--root',
                        default='../template/MNIST',
                        help='root dir to the project')
    parser_forget.add_argument('--expname',
                        default='EXP1',
                        help='name of exp, will affect the path and dataset splitting')
    parser_forget.add_argument('--teacher_model',
                        default='./models/EXP1/base/best_model.pth',
                        help='relative path of model to be distilled to the root')
    parser_forget.add_argument('--KD_label',
                        default='KD0.25',
                        help='the name of base dataset used for KD, should be defined in CONFIG')
    parser_forget.add_argument('--test_label',
                        default='TEST1',
                        help='label of the test data, defined in Dataset.py/Config')
    parser_forget.add_argument('--cal_label',
                        default='CAL1',
                        help='label of the calibration data, defined in Dataset.py/Config')
    parser_forget.add_argument('--cal_test_label',
                        default='CALTEST1',
                        help='label of the calibration test data, defined in Dataset.py/Config')
    parser_forget.add_argument('--query_label',
                        default='QO1',
                        help='label of the query data, defined in Dataset.py/Config, here the query dataset should overlap with training dataset')
    parser_forget.add_argument('--add_risk_loss',
                        type=int,
                        default=1,
                        help='1: will add risk loss when running KP, 0: same as pure KD')
    parser_forget.add_argument('--nclass',
                        type=int,
                        default=10,
                        help='number of classes')
    parser_forget.add_argument('--train_batch_size',
                        type=int,
                        default=32)
    parser_forget.add_argument('--KP_infer_batch_size',
                        type=int,
                        default=128,
                        help='batch size for inference during membership attack')
    parser_forget.add_argument('--device',
                        default='cuda:0')
    parser_forget.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='number of epochs')
    parser_forget.add_argument('--T',
                        type=float,
                        default=4.0,
                        help='temperature for ST')
    parser_forget.add_argument('--lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser_forget.add_argument('--lambda_kd',
                        type=float,
                        default=1,
                        help='trade-off parameter for kd loss')
    parser_forget.add_argument('--lambda_risk',
                        type=float,
                        default=10,
                        help='trade-off parameter for risk loss')
    parser_forget.add_argument('--num_workers',
                        type=int,
                        default=5,
                        help='number of num_workers')
    parser_forget.add_argument('--command_class',
                              default=1,
                              type=int,
                              help='for internal use only, no change')

    args = parser.parse_args()
    return args

def main(args):
    if args.command_class == 0:
        Audit.one_command_api(args)
    elif args.command_class == 1:
        Forget.one_command_api(args)

if __name__ == '__main__':
    args = parser()
    print(args)
    main(args)