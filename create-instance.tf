resource "google_compute_instance" "default" {
  name         = "vm-common-voice"
  machine_type = "n2-standard-2"
  zone         = var.def_region_zone

  boot_disk {
    initialize_params {
      image = "gce-uefi-images/ubuntu-1804-lts"
    }
  }

  network_interface {
    network = "default"

    access_config {
      // Include this section to give the VM an external ip address
    }
  }

  metadata_startup_script = file("startup.sh")

  // Apply the firewall rule to allow external IPs to access this instance
  tags = ["http-server"]

  metadata = {
    ssh-keys = "ubuntu:${file("~/.ssh/id_rsa.pub")}"
  }

  provisioner "file" {
    source      = "Dockerfile"
    destination = "/tmp/Dockerfile"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      timeout     = "500s"
      private_key = file("~/.ssh/id_rsa")
      host        = google_compute_instance.default.network_interface.0.access_config.0.nat_ip
    }
  }

  provisioner "file" {
    source      = "docker-compose.yml"
    destination = "/tmp/docker-compose.yml"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      timeout     = "500s"
      private_key = file("~/.ssh/id_rsa")
      host        = google_compute_instance.default.network_interface.0.access_config.0.nat_ip
    }
  }

  provisioner "file" {
    source      = "nginx_rev.conf"
    destination = "/tmp/nginx_rev.conf"

    connection {
      type        = "ssh"
      user        = "ubuntu"
      timeout     = "500s"
      private_key = file("~/.ssh/id_rsa")
      host        = google_compute_instance.default.network_interface.0.access_config.0.nat_ip
    }
  }


  provisioner "remote-exec" {
    connection {
      type        = "ssh"
      user        = "ubuntu"
      timeout     = "500s"
      host        = google_compute_instance.default.network_interface.0.access_config.0.nat_ip
      private_key = file("~/.ssh/id_rsa")
    }

    inline = [
      "sleep 30",
      "sudo apt-get update -y",
      "sudo apt-get install git -yq",
      "sudo curl -sSL https://get.docker.com/ | sh",
      "sudo usermod -aG docker `echo $USER`",
      "cd /opt && sudo git clone https://github.com/dachosen1/Common-Voice.git commvoice",
      "sudo curl -L \"https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose",
      "sudo chmod +x /usr/local/bin/docker-compose",
      "cd /opt/commvoice",
      "sudo mv /opt/commvoice/Dockerfile /opt/commvoice/Dockerfile.org",
      "sudo mv /opt/commvoice/docker-compose /opt/commvoice/docker-compose.org",
      "sudo mv /opt/commvoice/commonvoice/default.conf /opt/commvoice/commonvoice/default.conf.org",
      "sudo cp /tmp/Dockerfile /opt/commvoice/Dockerfile",
      "sudo cp /tmp/nginx_rev.conf /opt/commvoice/commonvoice/default.conf",
      "sudo cp /tmp/docker-compose.yml /opt/commvoice/docker-compose.yml",
      "sudo docker-compose build",
      "sudo docker-compose up -d",
      "sudo docker-compose ps"
    ]
  }
}

resource "google_compute_firewall" "http-server" {
  name    = "default-allow-http-terraform"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "22", "443"]
  }

  // Allow traffic from everywhere to instances with an http-server tag
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

output "ip" {
  value = google_compute_instance.default.network_interface.0.access_config.0.nat_ip
}
