import signal

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
    #signal.signal(signal.SIGALRM, signal.SIG_DFL)
    return ok
