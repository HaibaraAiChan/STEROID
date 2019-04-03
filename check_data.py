

import os

steroid = './data/positive'
other = './data/negative'
voxel_folder_positive = './voxel_output/positive'
voxel_folder = './voxel_output/'

list_steroid = []
list_other = []
pos_voxel_name_list = ['1e6wC_EST','1jglB_EST','2z77C_EST'
                        ,'3fzwA_EST','3m8cC_EST','3olsA_EST','4jvlB_EST']
neg_voxel_name_list = []

for foldername in os.listdir(voxel_folder):
    if 'voxel_output_' in foldername:
        for f in os.listdir(voxel_folder+foldername):
            neg_voxel_name_list.append(f[0:-4])

neg_voxel_name_list= list(set(neg_voxel_name_list))

with open(steroid) as ad_in:
    for line in ad_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')
        tmp1 = ttmp[0].split('.')

        aa = tmp1[0]

        res1 = any(aa in voxel for voxel in pos_voxel_name_list)

        if res1:
            list_steroid.append(aa)
        else:
            print aa
    list_steroid.sort()
    list_steroid = list(set(list_steroid))
    # print list_steroid
    print len(list_steroid)

ad_in.close()

with open(other) as ot_in:
    for line in ot_in.readlines():
        temp = line.replace(' ', '').replace('\n', '')
        ttmp = temp.split('\t')

        tmp1 = ttmp[0].split('.')

        aa = tmp1[0]

        res1 = any(aa in voxel for voxel in neg_voxel_name_list)

        if res1:
            list_other.append(aa)
        else:
            print aa

    list_other.sort()
    list_other = list(set(list_other))
    print len(list_other)
ot_in.close()

if os.path.exists("steroid"):
    os.remove("steroid")
with open("steroid", "w") as outf:
    for i in range(len(list_steroid)):
        outf.write('%s\n' % list_steroid[i])
outf.close()

if os.path.exists("other"):
    os.remove("other")
with open("other", "w") as outf:
    for i in range(len(list_other)):
        outf.write('%s\n' % list_other[i])
outf.close()
