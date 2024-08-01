import numpy as np 
from scipy import sparse
from scipy.sparse import diags
import re
from _CommunityDetect import CommunityDetect
import pickle


def pornhub():
    # Preprocess Data
    viewIdmapTags = dict()
    Tags = set()
    with open("./empirical_data/pornhub/porn-with-dates-2022.csv", 'r', encoding='utf-8') as fr:
        fr.readline()
        line = fr.readline()
        while line:
            sline = re.split(',"\[|]",', line.strip())
            viewId = sline[0].split(',')[1].split('=')[1]
            if len(sline) < 2:
                # some video have 1 tag
                # print(viewId)
                sline = re.split(',\[|],', line.strip())
            if viewId == "ph5bfe7bec849b0":
                # title have seperator
                tags = sline[2].replace('\'', '').split(', ')
            else:
                tags = sline[1].replace('\'', '').split(', ')
            for i, tag in enumerate(tags):
                if tag == '[Amateur':
                    tag = tag.replace('[', '')
                    tags[i] = tag
                if tag == '"[Babe':
                    tag = tag.replace('[', '')
                    tag = tag.replace('\"', '')
                    tags[i] = tag
                if tag == '':
                    pass
                if tag not in Tags:
                    Tags.add(tag)
            viewIdmapTags[viewId] = tags
            line = fr.readline()
            # print(f'id={viewId}, tag={tags}')
    videoNum = len(viewIdmapTags)
    tagNum = len(Tags)
    print(f'videos:{videoNum}, tags:{tagNum}')
    nodes_num = videoNum + tagNum
    data = []
    rowInx = []
    colInx = []
    # A = np.zeros((nodes_num, nodes_num))
    videoIdMaps = dict()
    tagIdMaps = dict()
    viewInx = 0
    tagInx = 0
    for vid in viewIdmapTags.keys():
        if vid not in videoIdMaps.keys():
            videoIdMaps[vid] = viewInx
            viewInx += 1
        for tag in viewIdmapTags[vid]:
            if tag not in tagIdMaps.keys():
                tagIdMaps[tag] = videoNum + tagInx
                tagInx += 1
            data.append(1)
            rowInx.append(videoIdMaps[vid])
            colInx.append(tagIdMaps[tag])
            # Undirected edge
            # data.append(1)
            # rowInx.append(tagIdMaps[tag])
            # colInx.append(videoIdMaps[vid])
    with open("./empirical_data/pornhub/videoIdmap.txt", 'w') as f:
        for vid in videoIdMaps.keys():
            f.write(f'{vid} {videoIdMaps[vid]}\n')
    with open("./empirical_data/pornhub/tagIdmap.txt", 'w') as f:
        for tag in tagIdMaps.keys():
            f.write(f'{tag} {tagIdMaps[tag]}\n')
    A = sparse.csr_array((data, (rowInx, colInx)), shape=(nodes_num, nodes_num), dtype=np.int8)
    A = A + A.T
    return A, videoNum, tagNum


def xHamster():
    # Preprocess Data
    viewIdmapTags = dict()
    Tags = set()
    with open("./empirical_data/xHamster/xhamster.csv", 'r', encoding='utf-8') as fr:
        fr.readline()
        line = fr.readline()
        while line:
            sline = re.split(',"\[|]",', line.strip())
            viewId = sline[0].split(',')[0]
            tags = None
            # if viewId.startswith("Dani"):
            #     print(viewId.isdigit())
            if viewId.isdigit():
                if len(sline) != 3:
                    # some video have 1 tag
                    # print(viewId)
                    sline = re.split(',\[|],', line.strip())
                if len(sline) < 2:
                    # some video no tag, no []
                    line = fr.readline()
                    # print(f'id={viewId}, tag={tags if tags is not None else "None"}')
                    continue
                tags = sline[1].replace('\'', '').split(', ')
                if len(tags) == 1 and tags[0] == '':
                    # some video no tag, but have []
                    line = fr.readline()
                    # print(f'id={viewId}, tag={tags if tags is not None else "None"}')
                    continue
                # if viewId == "ph5bfe7bec849b0":
                #     # title have seperator
                #     tags = sline[2].replace('\'', '').split(', ')
                # else:

                for i, tag in enumerate(tags):
                    # if tag == '[Amateur':
                    #     tag = tag.replace('[', '')
                    #     tags[i] = tag
                    # if tag == '"[Babe':
                    #     tag = tag.replace('[', '')
                    #     tag = tag.replace('\"', '')
                    #     tags[i] = tag
                    # if tag == '':
                    #     pass
                    if tag not in Tags:
                        Tags.add(tag)
                viewIdmapTags[viewId] = tags
            line = fr.readline()
            # print(f'id={viewId}, tag={tags if tags is not None else "None"}')
    videoNum = len(viewIdmapTags)
    tagNum = len(Tags)
    print(f'videos:{videoNum}, tags:{tagNum}')
    nodes_num = videoNum + tagNum
    data = []
    rowInx = []
    colInx = []
    # A = np.zeros((nodes_num, nodes_num))
    videoIdMaps = dict()
    tagIdMaps = dict()
    viewInx = 0
    tagInx = 0
    for vid in viewIdmapTags.keys():
        if vid not in videoIdMaps.keys():
            videoIdMaps[vid] = viewInx
            viewInx += 1
        for tag in viewIdmapTags[vid]:
            if tag not in tagIdMaps.keys():
                tagIdMaps[tag] = videoNum + tagInx
                tagInx += 1
            data.append(1)
            rowInx.append(videoIdMaps[vid])
            colInx.append(tagIdMaps[tag])
            # Undirected edge
            # data.append(1)
            # rowInx.append(tagIdMaps[tag])
            # colInx.append(videoIdMaps[vid])
    with open("./empirical_data/xHamster/videoIdmap.txt", 'w') as f:
        for vid in videoIdMaps.keys():
            f.write(f'{vid} {videoIdMaps[vid]}\n')
    with open("./empirical_data/xHamster/tagIdmap.txt", 'w') as f:
        for tag in tagIdMaps.keys():
            f.write(f'{tag} {tagIdMaps[tag]}\n')
    A = sparse.csr_array((data, (rowInx, colInx)), shape=(nodes_num, nodes_num), dtype=np.int8)
    A = A + A.T
    return A, videoNum, tagNum


def exp(name="pornhub"):
    if name == "pornhub":
        A, n1, n2 = pornhub()
    elif name == "xhamster":
        A, n1, n2 = xHamster()
    else:
        A, n1, n2 = None, None, None
    A_BHpartition, A_BHnumgroups = CommunityDetect(A).BetheHessian()
    print(f'BH_A Level 1 video nodes group: {np.unique(A_BHpartition[:n1])}')
    print(f'BH_A Level 1 tag nodes group: {np.unique(A_BHpartition[n1:])}')
    with open(f"./result/detectionEmpirical/{name}_A.pkl", "wb") as fw:
        pickle.dump(A_BHpartition, fw)

    AA = A.dot(A)
    AA = AA - diags(np.diag(AA.toarray()), 0)  # remove diagonal
    # AA_WBHpartition, AA_WBHnumgroups = CommunityDetect(AA).BetheHessian(weighted=True)
    # print(f'WBH_AA Level 1 video nodes group: {np.unique(AA_WBHpartition[:n1])}')
    # print(f'WBH_AA Level 1 tag nodes group: {np.unique(AA_WBHpartition[n1:])}')
    # with open(f"./result/detectionEmpirical/{name}_AA.pkl", "wb") as fw:
    #     pickle.dump(AA_WBHpartition, fw)

    BBT = AA[:n1, :n1]
    BBT_WBHpartition, BBT_WBHnumgroups = CommunityDetect(BBT).BetheHessian(weighted=True)
    print(f'WBH_BBT Level 1 video nodes group: {np.unique(BBT_WBHpartition)}')
    with open(f"./result/detectionEmpirical/{name}_BBT.pkl", "wb") as fw:
        pickle.dump(BBT_WBHpartition, fw)

    BTB = AA[n1:, n1:]
    BTB_WBHpartition, BTB_WBHnumgroups = CommunityDetect(BTB).BetheHessian(weighted=True)
    print(f'WBH_BTB Level 1 tag nodes group: {np.unique(BTB_WBHpartition)}')
    with open(f"./result/detectionEmpirical/{name}_BTB.pkl", "wb") as fw:
        pickle.dump(BTB_WBHpartition, fw)


if __name__ == '__main__':
    # exp()
    exp("xhamster")
