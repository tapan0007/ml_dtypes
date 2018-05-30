import os
import tempfile
import shutil
import sh

def copy_dir(src, dst):
    """ Helper to copy a directory tree.
    :param src: Source directory.
    :param dst: Destination directory.
    """
    from sh import cp
    import sh
    cp("-ar", sh.glob(src + '/*'), dst)
    cp("-ar", sh.glob(src + '/.*'), dst)


class KickstartIso(object):
    def __init__(self, iso_path, kickstart_file, delete_iso_dir=True):
        """ Create kickstart ISO based on given initial ISO.

        :param iso_path: Distro ISO location(only tested on ubuntu).
        :param kickstart_file: Kickstart file (auto-answer file for installation)
        :param delete_iso_dir: If set to False does not do cleanup (useful only during development of this tool).
        """
        self.iso_path = iso_path
        self.iso_dir = self._mount_iso()
        self.new_iso_dir = tempfile.mkdtemp(suffix='.iso.new')
        self.kickstart_file = kickstart_file
        self.delete_iso_dir = delete_iso_dir

    def __del__(self):
        if self.iso_dir:
            self.destroy()

    def destroy(self):
        if self.delete_iso_dir:
            sh.umount(self.iso_dir)
            shutil.rmtree(self.new_iso_dir)
        else:
            print(self.new_iso_dir, 'is not deleted.')
        self.iso_dir = None

    def _mount_iso(self):
        from sh import mount
        iso_dir = tempfile.mkdtemp(suffix='iso.vanilla')
        print("Mounting {} on {}".format(self.iso_path, iso_dir))
        mount("-oloop", self.iso_path, iso_dir)
        return iso_dir

    def build(self):
        """ Build new ISO
        :return: Path of the new ISO file.
        """
        copy_dir(self.iso_dir, self.new_iso_dir)

        # set lang as English
        with open(os.path.join(self.new_iso_dir, "isolinux", "lang"), "w") as f:
            f.write("en")

        kickstart_file_name = "ks-custom.cfg"
        # setup boot menu for kickstart and mark it as default.
        menu = ("default kickstart)\n" + \
                "  label kickstart\n" + \
                "  menu label ^Install Kickstarted:) Linux\n" + \
                "  kernel /install/vmlinuz\n" + \
                "  append  file=/cdrom/preseed/{ks_name} vga=788 initrd=/install/initrd.gz" + \
                " priority=critical locale=en_US\n").format(ks_name=kickstart_file_name)
        with open(os.path.join(self.new_iso_dir, "isolinux", "txt.cfg"), "r+") as f:
            old_content = f.readlines()
            f.seek(0)
            f.write(menu)
            f.writelines(old_content[1:])

        # set boot menu timeout to 1 second (0=forever)
        with open(os.path.join(self.new_iso_dir, "isolinux", "isolinux.cfg"), "r+") as f:
            lines = f.readlines()
            lines = [l.replace('timeout 0', 'timeout 1') for l in lines]
            f.seek(0)
            f.writelines(lines)

        #copy the kickstart file.
        shutil.copy(self.kickstart_file, os.path.join(self.new_iso_dir, "preseed", kickstart_file_name))

        return self._make_iso()

    def _make_iso(self):
        from sh import mkisofs
        from sh import isohybrid

        output_file = "/tmp/ubuntu-16.04-myowninstall-amd64.iso"
        mkisofs("-J", "-l", "-b", "isolinux/isolinux.bin", "-no-emul-boot",
                "-boot-load-size", 4, "-boot-info-table", "-z", "-iso-level", 4,
                "-c", "isolinux/isolinux.cat", "-o", output_file,
                "-joliet-long", self.new_iso_dir)
        isohybrid(output_file)
        return output_file