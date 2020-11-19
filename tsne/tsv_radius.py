import os
import sys

sys.path.append("../")

from libs.configs import cfgs


scr_tsv = './dcl_log/{}/dcl_meta.tsv'.format(cfgs.VERSION)
omega = 180 / 4


fr = open(scr_tsv, 'r')
lines = fr.readlines()
fr.close()

fw_tsv = open(os.path.join('dcl_log/{}'.format(cfgs.VERSION), 'dcl_meta_{}.tsv'.format(omega)), 'w')
# fw_tsv.write("Index\tLabel\n")
for ii, l in enumerate(lines):
    index = int(l.split('\t')[-1].split('\n')[0]) // (omega + 5e-5)
    # index = min(int(l.split('\t')[-1].split('\n')[0]) // radius, 89)
    fw_tsv.write("%.1f\n" % (index * omega))
fw_tsv.close()
