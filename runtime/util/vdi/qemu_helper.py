import os

def create_disk_image(size, path, vdi_name):
    """ Creates new QEMU disk image.

    :param size: Size of the new image in GB.
    :param path: Output directory.
    :return: Full path name of the new image.
    """
    from subprocess import Popen
    disk_image_path = os.path.join(path, vdi_name)
    print("Creating disk image {} of size ({}G)".format(disk_image_path, size))
    qemu_img_process = Popen(["qemu-img", "create", "-f", "qcow2", disk_image_path, "{}G".format(size)])
    qemu_img_process.communicate()
    return disk_image_path


def start_vm(hda, cdrom=None, qemu="qemu-system-x86_64", enable_kvm=True, vnc_port=1, tcp_forward_port=5555):
    """ Start QEMU VM and waits till it finishes.
    :param hda: Location of harddisk image.
    :param cdrom: Location of the ISO file.
    :param qemu: Qemu binary file full path.
    :param enable_kvm: If True enables KVM.
    :param vnc_port: VNC port number.
    :param tcp_forward_port: TCP forward port number.
    """
    import subprocess
    cmd = [qemu, '-m', "2048", '-hda', hda]
    if enable_kvm:
        cmd.append('--enable-kvm')
    cmd += ['-device', 'e1000,netdev=net0,mac=52:55:00:d1:55:01']
    cmd += ['-netdev', 'user,id=net0,hostfwd=tcp::{port}-:22'.format(port=tcp_forward_port)]
    cmd += ['-vnc', ':{port}'.format(port=vnc_port)]
    if cdrom:
        cmd += ['-cdrom', cdrom]

    #print(' '.join(cmd), flush=True)
    p = subprocess.Popen(cmd)
    p.communicate()
