variable "api_key" {}
variable "public_key_path" {}

terraform {
 required_providers {
    vastai = {
      source = "realnedsanders/vastai"
    }
  }
}

provider "vastai" {
  api_key = var.api_key
  api_url = "https://console.vast.ai"
}

resource "vastai_ssh_key" "my_key" {
   ssh_key = file(var.public_key_path)
}


# Search for an affordable RTX 4090 offer
data "vastai_gpu_offers" "rtx4090" {
  gpu_name           = "RTX 4090"
  num_gpus           = 1
  max_price_per_hour = 0.50
  limit              = 5
}

resource "vastai_instance" "training" {
  offer_id = data.vastai_gpu_offers.rtx4090.most_affordable.id
  disk_gb  = 50
  image    = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
  label    = "ml-training-run"
  ssh_key_ids = [vastai_ssh_key.my_key.id]
}
