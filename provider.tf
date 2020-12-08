provider "google" {
  project = "commvoice"
  region  = var.def_region
}

variable "def_region" {
  type    = string
  default = "us-west3"
}

variable "def_region_zone" {
  type    = string
  default = "us-west3-a"
}
