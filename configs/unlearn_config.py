import argparse

parser = argparse.ArgumentParser()
# Device
parser.add_argument("-gpu", type= bool, default= True, help= "use gpu or not")
# Dataset
parser.add_argument("-root", type= str, default= "./data", help= "Dataset root directory")
parser.add_argument("-dataset", type= str, help= "Dataset configuration",
                    choices=[
                        "Path",
                        "Derma",
                        "Retina",
                        "Blood"
                    ])
parser.add_argument("-img_size", type= int, default= 128, help= "Dataset image size",
                    choices= [28, 64, 128, 224])
# Model
parser.add_argument("-model_root", type= str, default= "./checkpoint", help= "Dataset root directory")
parser.add_argument("-model", type= str, help= "Model selection")
parser.add_argument("-save_model", type= bool, help= "Save trained model option")

# Unlearn configuration
parser.add_argument("-unlearn_method", type= str,
                    choices= ["baseline", "retrain", "fine_tune", "maverick"],
                    help= "Baselines unlearn method")

parser.add_argument("-scenario", type= str,
                    choices= ["class", "client", "sample"], help= "Training and unlearning scenario")

parser.add_argument("-model_path", type= str,
                    help= "Trained model path")
parser.add_argument("-unlearn_class", type= int, help= "Class to unlearn")
parser.add_argument("-sample_number", type= int, help= "Unlearn noise sampling number")
parser.add_argument("-min_sigma", type= float, help= "Minimum Gaussian noise standard deviation")
parser.add_argument("-max_sigma", type= float, help= "Maximum Gaussian noise standard deviation")
parser.add_argument("-mean", type= float, help= "Mean of Gaussian noise") # 0 for class unlearning
parser.add_argument("-lambda_coef", type= float, help= "Hyperparameter for KL divergence")

# Unlearn hyperparameter
parser.add_argument("-epochs", type= int, help= "Training epochs")
parser.add_argument("-batch_size", type= int, help= "Training batch size")
parser.add_argument("-lr", type=float, help='Learning rate')
parser.add_argument('-momentum', type=float, help='SGD momentum (default: 0.5)')

parser.add_argument("-num_clients", type= int, help= "Number of client participate")
parser.add_argument("-backdoor_size", type= int, help= "Backdoor pixel square size")

parser.add_argument('-report_training', type= bool, default= True, help= "option to show training performance")
parser.add_argument('-report_interval', type= int, default= 5, help= "training performance report interval")

# Set seed
parser.add_argument("-seed", type=int, help="Seed for runs")

arguments = parser.parse_args()
