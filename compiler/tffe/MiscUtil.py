import signal
import os
import stat

# Returns True opon success, False upon timeout
class ExecTimeout:
  @staticmethod
  def run(method, args, timeout):

    def handler(signum, frame):
      raise Exception("ExecTimeout Timeout")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    ok = False
    try:
      method(args)
      ok = True
    except:
      print("ERROR: Timeout of %d sec exceeded" % timeout)
    signal.alarm(0)
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    return ok


def remove_write_permissions(path):
  """Remove write permissions from all files in this path,
  while keeping all other permissions intact.

  Params:
      path:  The path whose permissions to alter.
  """

  for root, dirs, files in os.walk(path):
    for f in files:
      path_f = os.path.join(root, f)

      NO_USER_WRITING = ~stat.S_IWUSR
      NO_GROUP_WRITING = ~stat.S_IWGRP
      NO_OTHER_WRITING = ~stat.S_IWOTH
      NO_WRITING = NO_USER_WRITING & NO_GROUP_WRITING & NO_OTHER_WRITING

      current_permissions = stat.S_IMODE(os.lstat(path_f).st_mode)
      os.chmod(path_f, current_permissions & NO_WRITING)