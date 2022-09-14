
from collections import defaultdict
import torch as th
import numpy as np
def gettrainusercred(trainuserid,reuserid,uids,task):
    # trainuserid list  返回list
    limitime = 30000
    # (0, 60(1h), 120(2h), 240(4h), 480(8h), 720(12h), 1440(24h), 2160(36h), default: 3000 represe
    userlist = []
    userdic = defaultdict(list)
    maxuser = []
    maxuserid = []
    maxreply = []
    maxreplyid = []
    with open("H:/zxx/BiGCN/FakeNewsDetection"+"./dataset/"+task+"/"+task+".train", 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            arr = line.strip().split("\t")
            userdic[arr[0]].append(arr[-1])
            with open("H:/zxx/BiGCN/FakeNewsDetection"+"/dataset/"+task+"/tree/" + str(arr[1]) + ".txt", 'r', encoding='utf8') as fin2:
                fin2.readline()
                tempcount = 0
                tempuserdic = defaultdict(list)
                tempreplydic = defaultdict(list)
                for lin in fin2.readlines():
                    rep = eval(lin.strip().split("->")[1])
                    rep2 = eval(lin.strip().split("->")[0])
                    time = float(rep[-1])
                    if time < limitime:
                        # tempcount = tempcount + 1
                        userdic[str(rep[0])].append(arr[-1])
                        if str(rep[0]) not in tempuserdic:
                            tempuserdic[str(rep[0])].append(1)
                            tempcount = tempcount + 1
                        if str(rep[1]) not in tempreplydic:
                            tempreplydic[str(rep[1])].append(1)
                        if str(rep2[1]) not in tempreplydic:
                            tempreplydic[str(rep2[1])].append(1)
                maxuser.append(tempcount)
                maxuserid.append(arr[1])
                maxreply.append(len(tempreplydic))
                maxreplyid.append(arr[1])


    with open("H:/zxx/BiGCN/FakeNewsDetection/"+"dataset/"+task+"/"+task+".dev", 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            arr = line.strip().split("\t")
            userdic[arr[0]].append(arr[-1])
            with open("H:/zxx/BiGCN/FakeNewsDetection/"+"dataset/"+task+"/tree/" + str(arr[1]) + ".txt", 'r', encoding='utf8') as fin2:
                fin2.readline()
                tempcount = 0
                tempuserdic = defaultdict(list)
                tempreplydic = defaultdict(list)
                for lin in fin2.readlines():
                    rep = eval(lin.strip().split("->")[1])
                    rep2 = eval(lin.strip().split("->")[0])
                    time = float(rep[-1])
                    if time < limitime:

                        userdic[str(rep[0])].append(arr[-1])
                        if str(rep[0]) not in tempuserdic:
                            tempuserdic[str(rep[0])].append(1)
                            tempcount = tempcount + 1
                        if str(rep[1]) not in tempreplydic:
                             tempreplydic[str(rep[1])].append(1)
                        if str(rep2[1]) not in tempreplydic:
                            tempreplydic[str(rep2[1])].append(1)

                maxuser.append(tempcount)
                maxuserid.append(arr[1])
                maxreply.append(len(tempreplydic))
                maxreplyid.append(arr[1])
    with open("H:/zxx/BiGCN/FakeNewsDetection/"+"dataset/"+task+"/"+task+".test", 'r', encoding='utf8') as fin:
        for line in fin.readlines():
            arr = line.strip().split("\t")
            userdic[arr[0]].append(arr[-1])
            with open("H:/zxx/BiGCN/FakeNewsDetection/"+"dataset/"+task+"/tree/" + str(arr[1]) + ".txt", 'r', encoding='utf8') as fin2:
                fin2.readline()
                tempcount = 0
                tempuserdic = defaultdict(list)
                tempreplydic = defaultdict(list)
                for lin in fin2.readlines():
                    rep = eval(lin.strip().split("->")[1])
                    rep2 = eval(lin.strip().split("->")[0])
                    time = float(rep[-1])
                    if time < limitime:
                        userdic[str(rep[0])].append(arr[-1])
                        if str(rep[0]) not in tempuserdic:
                            tempuserdic[str(rep[0])].append(1)
                            tempcount = tempcount + 1
                        if str(rep[1]) not in tempreplydic:
                             tempreplydic[str(rep[1])].append(1)
                        if str(rep2[1]) not in tempreplydic:
                            tempreplydic[str(rep2[1])].append(1)

                maxuser.append(tempcount)
                maxuserid.append(arr[1])
                maxreply.append(len(tempreplydic))
                maxreplyid.append(arr[1])
    # t15 2964个用户  '552112474913136641'  214个帖子
    i = 0
    j = 0
    while maxuser[j] != 2812:
        j = j + 1
    while maxreplyid[i] != '614610920782888960':
        i = i + 1
    for k, v in userdic.items():
        positive = 0
        negative = 0

        for label in v:
            if label in ['non-rumor', 'true']:
                positive += 1
            else:
                negative += 1

        ulabel = -1
        if positive == 0:
            ulabel = 2
        elif negative == 0:
            ulabel = 0
        else:
            ulabel = 1
        userdic[k] = ulabel
    print('')

    # 找到dic中key相同的
    for i in trainuserid:
        for keyv in uids.items():
            if i == keyv[1]:
                userlist.append(userdic[keyv[0]])
                break
    reuserlist = []
    for i in reuserid:
        templist = []
        for reuserid in i:
            if reuserid == 0:
                templist.append(3)
            else:
                for keyv in uids.items():
                    if reuserid == keyv[1]:
                        templist.append(userdic[keyv[0]])
                        break
        reuserlist.append(templist)



    return userlist,reuserlist
y,_ = gettrainusercred([],[],[],'Twitter16')



