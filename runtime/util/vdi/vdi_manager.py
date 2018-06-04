#!/usr/bin/env python3

"""
    Create new VM image based on given ISO(tested on ubuntu ISOs).
    High level operation overview:
        1) Mount given ISO and copy all files and directories to new directory.
        2) Add boot menu for kickstart.
        3) Copy kick start file
        4) Create bootable ISO from the new directory.
        5) Create qemu disk image.
        6) Start qemu with new disk image and iso.
        7) Wait for qemu to finish.

    Requirements:
        pip3 install sh boto3
        sudo apt-get install syslinux-utils fuseiso

    Example usage:
        sudo ./vdi_manager.py create --iso-path ~/iso/ubuntu-16.04.4-server-amd64.iso --qemu /build/qemu_inkling/x86_64-softmmu/qemu-system-x86_64
            This will run for 5-10min and will create a new image with ubuntu 16.04.
            The sudo usage is required because `mount` is used in the script.
            Because of that the owner of the file has to be changed before uploading.
        sudo chown $USER vm_disk.qcow2
            Finally it can be uploaded to S3 using.
        ./vdi_manager.py upload --vdi-path vm_disk.qcow2
"""
import os
import argparse
import boto3
import botocore

from ks_iso import KickstartIso
import qemu_helper

DEFAULT_VDI_NAME = 'vm_disk.qcow2'
AWS_PROFILE_NAME = 'kaena'
S3_BUCKET_NAME = 'kaena-vdi'

def create_vdi(iso_path, kickstart_file, disk_size, output_path, qemu):
    iso = KickstartIso(iso_path, kickstart_file)
    iso_path = iso.build()

    disk_image_path = qemu_helper.create_disk_image(disk_size, path=output_path, vdi_name=DEFAULT_VDI_NAME)

    print("Installing new ISO on disk (This would take ~10 mins)")
    qemu_helper.start_vm(hda=disk_image_path, cdrom=iso_path, qemu=qemu)

    print("New VM disk image", disk_image_path)
    iso.destroy()


def list_vdi(aws_profile):
    s3, bucket = create_s3_session(aws_profile)
    print('VDI available in S3:')
    for object in bucket.objects.all():
        print(object)


def upload_vdi(aws_profile, vdi_path, s3name):
    s3, bucket = create_s3_session(aws_profile)
    bucket.upload_file(vdi_path, s3name)


def download_vdi(aws_profile, vdi_path, s3name):
    s3, bucket = create_s3_session(aws_profile)
    try:
        bucket.download_file(s3name, os.path.join(vdi_path, s3name))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("VDI does not exists in the S3.")
        else:
            raise


def create_s3_session(aws_profile):
    session = boto3.Session(profile_name=aws_profile)
    s3 = session.resource(service_name='s3', region_name='us-east-1')
    bucket = s3.Bucket(S3_BUCKET_NAME)
    return s3, bucket


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title="commands", dest='command')

    parser_create = subparsers.add_parser('create', help='Create VM Disk Image locally from kickstart file.')
    parser_create.add_argument("--iso-path", required=True, help="Path to ISO image")
    parser_create.add_argument("--disk-size", default=10, help="New VM's disk size in GB")
    parser_create.add_argument("--output-path", default='.', help="New VM's disk size in GB")
    parser_create.add_argument("--kickstart-file", default='ubuntu-vm-minimal.seed', help="Kickstart file.")
    parser_create.add_argument("--qemu", help="Full path to QEMU binary")

    parser_upload = subparsers.add_parser('upload', help='Upload VM Disk Image to S3.')
    parser_upload.add_argument("--vdi-path", required=True, help="Path to VDI which needs to uploaded.")
    parser_upload.add_argument("--name", default=DEFAULT_VDI_NAME, help="Name to use in S3.")

    parser_download = subparsers.add_parser('download', help='Download VM Disk Image from S3.')
    parser_download.add_argument("--vdi-path", default='.', help="Path to store ISO image.")
    parser_download.add_argument("--name", default=DEFAULT_VDI_NAME, help="Name of the image to download")

    parser_list = subparsers.add_parser('list', help='List VM Disk Images available in the S3.')

    parser.add_argument('--aws-profile', help='AWS Profile Name', default=AWS_PROFILE_NAME)
    args = parser.parse_args()

    if args.command == 'create':
        create_vdi(args.iso_path, args.kickstart_file, args.disk_size, args.output_path, args.qemu)
    elif args.command == 'list':
        list_vdi(args.aws_profile)
    elif args.command == 'upload':
        upload_vdi(args.aws_profile, vdi_path=args.vdi_path, s3name=args.name)
    elif args.command == 'download':
        download_vdi(args.aws_profile, vdi_path=args.vdi_path, s3name=args.name)
    else:
        parser.print_help()


# Entry point
main()
