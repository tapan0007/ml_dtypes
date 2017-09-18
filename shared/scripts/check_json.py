#!/usr/bin/python
import json
import sys

for json_name in sys.argv[1:]:
    with open(json_name) as json_file:
        print "checking %s" % (json_name)
        try:
            json_load = json.load(json_file)
        except Exception,e:
            print("Unexpected error:", e)
            sys.exit(255)

