grep -e " @ " -e "_event_" -e "sync_ready=1" -e Execute -e SetEvent -e ClearEvent $1 | sed 's/ *\[[0-9]*, *[0-9]*\] *: */=/' > $1.tmp
ex -s "+%s/\(event_mode.*\)\n/\1/" "+%s/ *\[\d, *\d\]: */=/" "+wq" $1.tmp
grep -v "event_mode *\[\d, *\d\]: 0x0" $1.tmp
