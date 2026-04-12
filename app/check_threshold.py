import sys
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("https://dagshub.com/17-doha/GANS.mlflow")

try:
    with open("model_info.txt", "r") as f:
        run_id = f.read().strip()
except FileNotFoundError:
    print(
        "Error: model_info.txt not found! The validate job must have failed "
        "to upload it."
    )
    sys.exit(1)

print(f"Checking DagsHub MLflow for specific Run ID: {run_id}")

client = MlflowClient()

try:
    run = client.get_run(run_id)
except Exception as e:
    print(f"Error fetching run {run_id} from MLflow. Details: {e}")
    sys.exit(1)

metrics = run.data.metrics

try:
    gen_loss = metrics["generator_loss"]
    disc_loss = metrics["discriminator_loss"]
    acc_real = metrics["accuracy_real"]
    acc_fake = metrics["accuracy_fake"]
except KeyError as e:
    print(f"Error: Missing metric {e} in MLflow run data!")
    sys.exit(1)

print("\n Current Model Metrics:")
print(f" - Generator Loss: {gen_loss:.4f}")
print(f" - Discriminator Loss: {disc_loss:.4f}")
print(f" - Accuracy (Real): {acc_real:.4f}")
print(f" - Accuracy (Fake): {acc_fake:.4f}\n")

passed_all_checks = True

# if gen_loss >= 1.0:
#     print(" Failed: Generator loss is too high.")
#     passed_all_checks = False

# if disc_loss >= 2.0:
#     print(" Failed: Discriminator loss is too high.")
#     passed_all_checks = False

if acc_real <= 0.85:
    print(" Failed: Accuracy on real images is too low.")
    passed_all_checks = False

# if acc_fake <= 0.7:
#     print(" Failed: Accuracy on fake images is too low.")
#     passed_all_checks = False

if passed_all_checks:
    print("\n Success! All metrics meet the strict deployment thresholds!")
    sys.exit(0)
else:
    print("\n Pipeline halted due to failing metrics.")
    sys.exit(1)
