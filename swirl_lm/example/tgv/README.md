# Swirl-LM Demo on Cloud TPU

This is an example of a stand-alone python script that uses the Swirl-LM
library. It demonstrates how to run a Taylor-Green Vortex Flow on Google
Cloud TPUs.

1. Check / set project:

   Check if your GC project is set on your local machine:

   ```sh
   gcloud config list
   ```

   If the output does not contain the correct project, then set it with:

   ```sh
   gcloud config set project <PROJECT>
   ```

1. Create the TPU nodes and VM:

   ```sh
   TPU=swirl-lm-demo
   ZONE=europe-west4-a
   gcloud compute tpus execution-groups create \
     --zone="$ZONE" --name="$TPU" --accelerator-type=v3-32 \
     --tf-version=2.9.1
   ```

   This step creates a slice of 32 TPUs (v3) and a GCE VM. The TPU hosts and
   the VM are automatically configured to be able to communicate. The VM image
   includes tensorflow.

   See https://cloud.google.com/tpu/docs/regions-zones for regions and TPU
   configurations. For most up-to-date info, use:

   ```sh
   gcloud compute tpus accelerator-types list --zone="$ZONE"
   ```

   Note: The VM and the TPU nodes have the same name. The name refers to the VM
   normally, and to the TPU cluster when used in the TPU APIs.

   Note: you might need to enable the TPU API for your project if it's not
   already enabled.

1. SSH into the VM:

   The previous command (execution-groups create) may automatically ssh into
   the VM. If not, ssh into the VM:

   ```sh
   gcloud compute ssh --zone="$ZONE" "$TPU"
   ```

   Run the next set of steps on the VM.

1. (VM) Clone Swirl LM from github:

   ```sh
   git clone https://github.com/google-research/swirl-lm.git
   ```

1. (VM) Install protobuf compiler, and if you plan to edit files, your favorite
editor:

   ```sh
   sudo apt-get install protobuf-compiler <editor>  # e.g., <editor>=emacs-nox
   ```

1. (VM) Install Swirl LM.

   ```sh
   pip install ./swirl-lm
   ```

   Note: this uses the files in the local repo and not in github; and also
   installs the package in the user directory and not in site-packages.

1. (VM) Run Swirl LM's set up script.

   ```sh
   bash ./swirl-lm/swirl_lm/example/tgv/setup.sh
   ```

   This script compiles proto files.

1. (VM) Run the solver.

   ```sh
   python3 ./swirl-lm/swirl_lm/example/tgv/main.py \
     --cx=1 --cy=1 --cz=8 \
     --data_dump_prefix=./data/tgv --data_load_prefix=./data/tgv \
     --config_filepath=./swirl-lm/swirl_lm/example/tgv/tgv_dns_nu2e3_piter10_quick.textpb \
     --num_steps=2000 --nx=128 --ny=128 --nz=6 --kernel_size=16 \
     --halo_width=2 --lx=99193.5483871 --ly=99193.5483871 --lz=99218.75 \
     --num_boundary_points=0 --dt=0.1 --u_mag=1.0 --p_ref=0.0 --rho_ref=1.0 \
     --project=<PROJECT> --tpu=<TPU> --zone=<ZONE> --output=./test.png
   ```

1. (VM) Check that the output file has been created.

   ```sh
   ls -l test.png
   ```

   Run the remaining commands are on your local machine and not on the VM.

1. Copy the output out of the VM to view it locally.

   ```sh
   gcloud compute scp --zone=$ZONE $TPU:test.png /tmp
   ```

1. Delete the TPU nodes and VM.

   ```sh
   gcloud compute tpus execution-groups delete --zone=$ZONE $TPU
   ```

   Note: This deletes both the TPUs and the VM. Deleting the VM also deletes
   its disk by default, so you will lose the cloned repo, etc. If you plan to
   re-run, you can delete only the TPUs by passing `--tpu-only``` to the
   command above; and later create only the TPUs by again passing the same
   flag.
