import sys
import os
import glob
import argparse
import shutil
import tarfile
import time
import boto3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNALL_PATH = '$KAENA_PATH/test/e2e'
# replication temporarily disabled for these tests
TEST_10K = ['7-rn50_nne_fp16_b4_wave-no_repl', '7-rn50_nne_fp16_wave-no_repl', '8-rn50_nne_fp16_b16_wave-no_repl']

parser = argparse.ArgumentParser()
parser.add_argument('--repo-name', help='repo name', default=None)
parser.add_argument('--aws-profile', help='name of aws credentials profile to use', default='kaena')
parser.add_argument('--bucket', help='output bucket name', default='kaena-test-bucket')
args = parser.parse_args()

session = boto3.Session(profile_name = args.aws_profile)
s3 = session.resource(service_name='s3', region_name='us-east-1')
bucket = s3.Bucket(args.bucket)

# get latest repo tar file from s3
repo_name = args.repo_name
if repo_name is None:
    repo = bucket.objects.filter(Prefix='kaena-')
    get_latest = lambda obj: int(obj.last_modified.strftime('%s'))
    for obj in sorted(repo, key=get_latest, reverse=True):
        repo_name = obj.key
        break

date = time.strftime("%Y%m%d")


def gen_hist_graph(kelf):
    """ Generate graph of history of 10K test for specific kelf.
    :param kelf: name of kelf to get history for
    :return:
    """
    os.mkdir("histtmp")
    hist_files = list(bucket.objects.filter(Prefix='dailies/history/%s/' % kelf))
    for f in hist_files[1:]:
        f_base = os.path.basename(f.key)
        bucket.download_file(f.key, "histtmp/%s" % f_base)

    # format: [ [top 1 match, top 5 match, within tolerance, nobs, min, max, mean, variance, skewness, kurtosis], ... ]
    y_goldens = []
    # format: [ [top 1 match, top 5 match], ... ]
    y_truths = []

    # create list of data for golden and truth in order by date (ascending)
    # naming for the files is: YYYYMMDD_testname.csv so [0:8] is the date.
    def sortby_date(csv_name):
        return int(os.path.basename(csv_name)[0:8])

    hist_csvs = sorted(glob.glob("histtmp/**.csv"), key=sortby_date)

    # limit number of data points in graph to 20 to avoid clutter
    if len(hist_csvs) > 20:
        hist_csvs = hist_csvs[-20:]
    # parse csv file created from result processing step below
    for hist in hist_csvs:
        golden_data = []
        truth_data = []
        with open(hist, 'r') as h:
            lines = h.read().splitlines()
            golden_data = lines[1].split(",")[2:]
            for i in range(len(golden_data)):
                try:
                    gdata = golden_data[i]
                    golden_data[i] = gdata[gdata.find("=")+1:]
                except:
                    pass
                if not golden_data[i][0].isdigit():
                    golden_data[i] = golden_data[i][1:]
                if not golden_data[i][-1].isdigit():
                    golden_data[i] = golden_data[i][:-1]
                golden_data[i] = golden_data[i].strip()

            truth_data = lines[2].split(",")[2:]
            for i in range(len(truth_data)):
                try:
                    tdata = truth_data[i]
                    truth_data[i] = tdata[tdata.find("=")+1:]
                except:
                    pass
                truth_data[i] = truth_data[i].strip()
        y_goldens.append(golden_data)
        y_truths.append(truth_data)
    x_points = [os.path.basename(fi)[0:8] for fi in hist_csvs]

    # plot top 1 and top 5 matches for golden and true (with annotations)
    def annotate_graph(axis, x, y):
        for xv, yv in zip(x,y):
            if yv != None:
                valLabel = "%.1f" % yv
                axis.annotate(valLabel, xy=(xv, yv), xytext=(-20,20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='lightyellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc,rad=0'))

    fig, ax = plt.subplots(1, sharex=True, figsize=(12,8))
    ax.set_xticks([x for x in range(len(x_points))])
    ax.set_xticklabels(x_points, rotation=90)
    ax.plot(x_points, [float(yg1[0]) for yg1 in y_goldens], marker='o', ls='solid', color='y', label='Golden Top 1 Match')
    annotate_graph(ax, x_points, [float(yg1[0]) for yg1 in y_goldens])
    ax.plot(x_points, [float(yg5[1]) for yg5 in y_goldens], marker='o', ls='dashed', color='y', label='Golden Top 5 Match')
    annotate_graph(ax, x_points, [float(yg5[1]) for yg5 in y_goldens])
    ax.plot(x_points, [float(yt1[0]) for yt1 in y_truths], marker='o', ls='solid', color='r', label='True Label Top 1 Match')
    annotate_graph(ax, x_points, [float(yt1[0]) for yt1 in y_truths])
    ax.plot(x_points, [float(yt5[1]) for yt5 in y_truths], marker='o', ls='dashed', color='r', label='True Label Top 5 Match')
    annotate_graph(ax, x_points, [float(yt5[1]) for yt5 in y_truths])
    ax.legend()
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of matches")
    ax.set_title("10K Test Match History - %s" % kelf)
    # TEMPORARY: since we only use 160 images
    ax.set_ylim(bottom=0, top=160)
    #ax.set_ylim(bottom=0, top=10000)
    plt.tight_layout()
    plt.savefig("10KTestMatch_%s.svg" % kelf)
    plt.savefig(os.path.splitext("10KTestMatch_%s.svg" % kelf)[0] + ".png")

    # upload updated graph and clean up
    bucket.upload_file("10KTestMatch_%s.svg" % kelf, "dailies/history/%s/10KTestMatch_%s.svg" % (kelf, kelf))
    bucket.upload_file("10KTestMatch_%s.png" % kelf, "dailies/history/%s/10KTestMatch_%s.png" % (kelf, kelf))
    shutil.rmtree('histtmp')


def main():
    # generate selected kelfs (specified in TEST_10K) using RunAll
    gen_kelf_cmd = "RunAll --test"
    for test in TEST_10K:
        gen_kelf_cmd += ' %s' % test
    os.mkdir('kelfs')
    os.chdir('kelfs')
    os.system('%s/%s' % (RUNALL_PATH, gen_kelf_cmd))
    os.chdir('..')

    # submit inference to batch for selected kelfs
    try:
        os.remove('inference_info.txt')
    except:
        pass

    for kelf in TEST_10K:
        print("*** %s" % kelf)
        try:
            batch_size = 1
            if '_b4_' in kelf:
                batch_size = 4
            elif '_b16_' in kelf or '_host' in kelf:
                batch_size = 16
            # TEMPORARY: use only 160 images to test on Jenkins
            # wait with no logs to avoid ThrottlingError when attempting to write to logs for 10K images
            submit_inf_cmd = "submit_inference --aws-profile=%s --kelf-dir=kelfs/%s --repo-name=%s --output-bucket=%s --input-file-limit=160 --input-batch-size=%s --wait-no-logs" % (args.aws_profile, kelf, repo_name, args.bucket, batch_size)
            os.system('./%s' % submit_inf_cmd)
        except:
            with open('batch-fail.txt', 'w') as log:
                log.write("Problem submitting inference (%s)." % kelf)
            sys.exit(-1)

    # process results
    tests = []
    with open('inference_info.txt', 'r') as info:
        tests = info.read().splitlines()
    for test in tests:
        test_name = test.strip().split(';')[0]
        testid = test.strip().split(';')[1]
        golden = "golden_fp32_p3"
        if 'fp16' in testid:
            golden = "golden_fp16_p3"
        try:
            print('************** %s **************' % testid)
            gen_csv_cmd = "rn50_generate_csv --testid=%s --aws-profile=%s --bucket=%s" % (testid, args.aws_profile, args.bucket)
            print("### Generating csv...")
            os.system('./%s' % gen_csv_cmd)
            report_diff_cmd = "rn50_report_diff --new=%s --golden=%s --aws-profile=%s --bucket=%s" % (testid, golden, args.aws_profile, args.bucket)
            print("### Comparing against golden (%s)..." % golden)
            os.system('./%s' % report_diff_cmd)
            report_vs_truth_cmd = "rn50_report_vs_truth --label_bucket=%s --new=%s --aws-profile=%s --bucket=%s --quiet" % ("kaena-imagenet-data-labels", testid, args.aws_profile, args.bucket)
            print("### Comparing against true labels...")
            os.system('./%s' % report_vs_truth_cmd)
        except:
            with open('batch-fail.txt', 'w') as log:
                log.write("Problem processing test results (%s)." % test_name)
            sys.exit(-1)

        # gather results files
        print("### Tar-ing result files...")
        with tarfile.open("%s.tar.gz" % testid, "w:gz") as test_results:
            test_results.add("/tmp/results_%s/report.csv" % testid, arcname="%s_report.csv" % testid)
            test_results.add("/tmp/diff_stats_%s_%s" % (testid, golden), arcname="golden_diff_%s_%s.csv" % (testid, golden))
            test_results.add("/tmp/diff_stats_%s.txt" % testid, arcname="report_vs_truth_%s.txt" % testid)

        # condense results into one file and add to history
        # files named YYYYMMDD_testname.csv, separated by each day
        print("### Adding new data to history...")
        history_file = "%s_%s.csv" % (date, test_name)

        gold = []
        truth = []
        with open("/tmp/diff_stats_%s_%s" % (testid, golden), 'r') as goldens:
            gold = goldens.read().splitlines()
        with open("/tmp/diff_stats_%s.txt" % (testid), 'r') as truths:
            truth = truths.read().splitlines()
        with open(history_file, 'w') as history:
            history.write("%s,   , %s\n" % (date, gold[1]))
            history.write("  , %s, %s, %s, %s, %s\n" % ("vs. Golden", gold[2], gold[3], gold[4], gold[5]))
            history.write("  , %s, %s, %s\n" % ("vs. Truth", truth[2], truth[3]))
        try:
            bucket.upload_file(history_file, "dailies/history/%s/%s" % (test_name, history_file))
        except:
            s3.create_bucket(Bucket="kaena-test-bucket/dailies/history/%s" % test_name)
            bucket.upload_file(history_file, "dailies/history/%s/%s" % (test_name, history_file))

        # update history graph
        print("### Updating history graph...")
        gen_hist_graph(test_name)

    # tar all test results and upload to S3
    results_file = "%s_results.tar.gz" % date
    with tarfile.open(results_file, "w:gz") as results:
        for indiv_results in glob.glob("*.tar.gz"):
            results.add(indiv_results)
    bucket.upload_file(results_file, "dailies/%s" % results_file)
    print(">>>> Uploaded results to '%s/dailies' (%s) <<<<" % (args.bucket, results_file))

    # cleanup
    shutil.rmtree('kelfs')
    for tar in glob.glob("*.tar.gz"):
        os.remove(tar)

if __name__ == '__main__':
    main()
    
