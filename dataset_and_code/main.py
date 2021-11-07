# Written by Mostafa Elyasi
import PGM_h
from numpy.linalg import norm
from scipy import optimize as optimize
from math import exp
from math import log

k_features = 6
landa_init = [0.30303594, 0.30279514, 0.25452369, 0.30318327, 0.28723748, 0.27421734]  # [1, 1, 1, 1, 1, 1]
# landa for 1000 train query [ 0.21909945  0.21877479  0.21689562  0.21909932  0.21211461  0.21429248] 90sec
# landa for 6000 train query [ 0.30303594  0.30279514  0.25452369  0.30318327  0.28723748  0.27421734] 310sec (5k3m)
# landa for 6000 train query [ 0.2295376   0.22583204  0.19447061  0.224045    0.23101649  0.19441793] 260sec 64k10m
# ----------------------------------------Read from Learning Model File Probabilities
with open('2gram64k10m.arpa', 'r') as f:  
    d = {}
    l = f.read().split('\n')  # Split using end line
    for ieach_char in l:
        values = ieach_char.split('\t')  # Split using 'tab'
        d[values[1]] = values[0]
    lang_model_ = d
    f.close()

with open('1gram_train64k.txt', 'r') as f:  
    freq = {}
    l = f.read().split('\n')  
    for count_1g in l:
        values = count_1g.split('\t')  
        freq[values[1]] = values[0]
    lang_model_1g = freq
    f.close()

with open('temp_train.txt', 'r') as f: 
    listC = []
    xi = []
    yi = []
    oi = []
    l = f.read().split('\n')  
    for count in l:
        values = count.split(',')  # Split using ','
        xi.append(values[0])
        yi.append(values[1])
        oi.append(values[2])
        listC.append(values)
    f.close()
Train_Data = listC


# ----------------------------------------f Function
def f(vay_pre, vay, language_model):
    vay_pre = vay_pre.lower()
    vay = vay.lower()
    yi_pre_and_yi = vay_pre + ' ' + vay
    if yi_pre_and_yi in language_model.keys():
        ix = language_model[yi_pre_and_yi]
        pro = float(ix)
        return pro
    else:
        return -99


# ----------------------------------------


def make_list(operation, x_i, length_of_check):
    temp_list = []
    list_of_char = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
                    'u', 'v', 'w', 'x', 'y', 'z']
    if operation == "nothing":
        temp_list.append(x_i)
        return temp_list
    elif operation == "del":
        for k in range(len(x_i)):
            for j in range(1, length_of_check + 1):
                first_part = x_i[:k]  #
                second_part = x_i[k + j:]  #
                temp_string = first_part + second_part
                temp_list.append(temp_string)
        return temp_list
    elif operation == "ins":
        for k in range(len(x_i)):
            for j in list_of_char:
                first_part = x_i[:k]
                second_part = x_i[k:]
                temp_string = first_part + j + second_part
                temp_list.append(temp_string)
        return temp_list
    elif operation == "subs":
        for k in range(len(x_i)):
            for j in list_of_char:
                if j != x_i[k]:
                    first_part = x_i[:k]
                    second_part = x_i[k + 1:]
                    temp_string = first_part + j + second_part
                    temp_list.append(temp_string)
        return temp_list
    elif operation == "trans":
        for k in range(len(x_i) - 1):
            temp1 = x_i[k]
            first_part = x_i[:k]
            for j in range(k + 1, len(x_i)):
                temp2 = x_i[j]
                second_part = x_i[k + 1:j]
                third_part = x_i[j + 1:]
                temp_string = first_part + temp2 + second_part + temp1 + third_part
                temp_list.append(temp_string)
        return temp_list
    elif operation == "merge" and x_i.find(" ") != -1:
        x_i = x_i.replace(" ", "-")
        temp_list.append(x_i)
        return temp_list
    elif operation == "split":
        for k in range(1, len(x_i)):
            first_part = x_i[:k]
            second_part = x_i[k:]
            temp_string = first_part + "-" + second_part
            temp_list.append(temp_string)
        return temp_list
    return ""


# ------------------------------------------------------------------------------------------


def pr_function(landa_tmp):
    pri = 0
    file_length = len(Train_Data)
    for i in range(file_length):
        yi_pre = "<s>"
        x = xi[i].split(" ")
        y = yi[i].split(" ")
        o = oi[i].strip('\n')
        o = o.split(" ")
        tmp_pri = pro_yox(x, yi_pre, y, o, landa_tmp, lang_model_, k_features, 1)
        pri += tmp_pri
    pri = norm(landa_tmp) ** 2 * 2 - log(pri, 10)
    return pri


def pre_fi_f(vay_pre, vay, language_model, landa_tmp, features):
    result = 0
    for i in range(features):
        result += landa_tmp[i] * f(vay_pre, vay, language_model)
    return result


def pre_fi_h(vay, ou, ix, length_of_check, landa_tmp, features):
    result = 0
    for i in range(features):
        result += landa_tmp[i] * PGM_h.h(vay, ou, ix, length_of_check, i)
    return result


def pro_yox(x, yi_pre, y, o, landa_tmp, language_model, features, idx):
    result = 0
    z = 1 / (1 ** 1)
    if idx == 0:
        xij = x
        yij = y
        oij = o
        if oij == "split":
            yij = yij.replace("-", " ")
            yij_temp = yij.split(" ")
            yi_pre = yij_temp[0]
            yij = yij_temp[1]
        elif oij == "merge":
            xij = xij.replace("-", " ")
        lfk = pre_fi_f(yi_pre, yij, language_model, landa_tmp, features)
        lhk = pre_fi_h(yij, oij, xij, 1, landa_tmp, features)
        pro_tmp = lfk + lhk
        result += pro_tmp
    else:
        length = len(o)
        for ocount in range(length):
            xij = x[ocount]
            yij = y[ocount]
            oij = o[ocount]
            if oij == "split":
                yij = yij.replace("-", " ")
                yij_temp = yij.split(" ")
                yi_pre = yij_temp[0]
                yij = yij_temp[1]
            elif oij == "merge":
                xij = xij.replace("-", " ")
            lfk = pre_fi_f(yi_pre, yij, language_model, landa_tmp, features)
            lhk = pre_fi_h(yij, oij, xij, 1, landa_tmp, features)
            pro_tmp = lfk + lhk
            result += pro_tmp
            yi_pre = y[ocount]
    if result > 500:
        result = 500
    result = exp(result)
    result /= z
    return result


# ------------------------------------------------------------------------------------------


def test_model(x, landa_tmp): 
    o1_list = ["nothing", "del", "subs", "ins", "trans"]
    count_x = x.count(" ") + 1
    x = x.split(" ")
    best_y1 = []
    best_y2 = []
    best_y3 = []
    best_y = []
    bfirst_oi = []
    bsecond_oi = []
    bthird_oi = []
    bfourth_oi = []
    grades = []
    yi_pre = "<s>"
    in_merge = 0
    for i in range(count_x):
        if in_merge:
            in_merge = 0
            continue
        xip1 = '<\s>'
        if i != count_x - 1:
            xip1 = x[i + 1]
        in_phrase = 0
        in_split = 0
        x_i = x[i]
        temp = ['nothing', x_i, in_split, in_merge, -99, -99, -99]
        temp1 = ['nothing', x_i, in_split, in_merge, -99, -99, -99]
        temp2 = ['nothing', x_i, in_split, in_merge, -99, -99, -99]
        temp3 = ['nothing', x_i, in_split, in_merge, -99, -99, -99]
        temp4 = ['nothing', x_i, in_split, in_merge, -99, -99, -99]
        temp5 = ['nothing', x_i, in_split, in_merge, -99, -99, -99]

        temp = f_err_correction(o1_list, yi_pre, temp, landa_tmp)
        first_ois = [temp[0]]
        first_yis = [temp[1]]
        temp = f_split(temp, landa_tmp)
        second_ois = [temp[0]]
        second_yis = [temp[1]]
        temp = f_merge(xip1, temp, landa_tmp)
        third_ois = [temp[0]]
        third_yis = [temp[1]]
        in_spliting = [temp[2]]
        in_merging = [temp[3]]
        best_g = [temp[4] + temp[5] + temp[6]]

        temp = f_split(temp1, landa_tmp)
        first_ois.append(temp[0])
        first_yis.append(temp[1])
        temp = f_merge(xip1, temp, landa_tmp)
        second_ois.append(temp[0])
        second_yis.append(temp[1])
        temp = f_err_correction(o1_list, yi_pre, temp, landa_tmp)
        third_ois.append(temp[0])
        third_yis.append(temp[1])
        in_spliting.append(temp[2])
        in_merging.append(temp[3])
        best_g.append(temp[4] + temp[5] + temp[6])

        temp = f_err_correction(o1_list, yi_pre, temp2, landa_tmp)
        first_ois.append(temp[0])
        first_yis.append(temp[1])
        temp = f_merge(xip1, temp, landa_tmp)
        second_ois.append(temp[0])
        second_yis.append(temp[1])
        temp = f_split(temp, landa_tmp)
        third_ois.append(temp[0])
        third_yis.append(temp[1])
        in_spliting.append(temp[2])
        in_merging.append(temp[3])
        best_g.append(temp[4] + temp[5] + temp[6])

        temp = f_merge(xip1, temp3, landa_tmp)
        first_ois.append(temp[0])
        first_yis.append(temp[1])
        temp = f_split(temp, landa_tmp)
        second_ois.append(temp[0])
        second_yis.append(temp[1])
        temp = f_err_correction(o1_list, yi_pre, temp, landa_tmp)
        third_ois.append(temp[0])
        third_yis.append(temp[1])
        in_spliting.append(temp[2])
        in_merging.append(temp[3])
        best_g.append(temp[4] + temp[5] + temp[6])

        temp = f_merge(xip1, temp4, landa_tmp)
        first_ois.append(temp[0])
        first_yis.append(temp[1])
        temp = f_err_correction(o1_list, yi_pre, temp, landa_tmp)
        second_ois.append(temp[0])
        second_yis.append(temp[1])
        temp = f_split(temp, landa_tmp)
        third_ois.append(temp[0])
        third_yis.append(temp[1])
        in_spliting.append(temp[2])
        in_merging.append(temp[3])
        best_g.append(temp[4] + temp[5] + temp[6])

        temp = f_split(temp5, landa_tmp)
        first_ois.append(temp[0])
        first_yis.append(temp[1])
        temp = f_err_correction(o1_list, yi_pre, temp, landa_tmp)
        second_ois.append(temp[0])
        second_yis.append(temp[1])
        temp = f_merge(xip1, temp, landa_tmp)
        third_ois.append(temp[0])
        third_yis.append(temp[1])
        in_spliting.append(temp[2])
        in_merging.append(temp[3])
        best_g.append(temp[4] + temp[5] + temp[6])

        bg = 0
        if best_g[3] >= best_g[1] and best_g[3] >= best_g[2] and best_g[3] >= best_g[0] and best_g[3] >= best_g[4] and \
                        best_g[3] >= best_g[5]:
            bg = 3
        elif best_g[1] >= best_g[0] and best_g[1] >= best_g[2] and best_g[1] >= best_g[3] and best_g[1] >= best_g[4] and \
                        best_g[1] >= best_g[5]:
            bg = 1
        elif best_g[2] >= best_g[1] and best_g[2] >= best_g[0] and best_g[2] >= best_g[3] and best_g[2] >= best_g[4] and \
                        best_g[2] >= best_g[5]:
            bg = 2
        elif best_g[4] >= best_g[0] and best_g[4] >= best_g[2] and best_g[4] >= best_g[3] and best_g[4] >= best_g[1] and \
                        best_g[4] >= best_g[5]:
            bg = 4
        elif best_g[5] >= best_g[1] and best_g[5] >= best_g[0] and best_g[5] >= best_g[3] and best_g[5] >= best_g[2] and \
                        best_g[5] >= best_g[4]:
            bg = 5

        first_oi = first_ois[bg]
        first_yi = first_yis[bg]
        second_oi = second_ois[bg]
        second_yi = second_yis[bg]
        third_oi = third_ois[bg]
        third_yi = third_yis[bg]
        in_split = in_spliting[bg]
        in_merge = in_merging[bg]

        x_i = third_yi
        fourth_yi = x_i
        fourth_oi = 'nothing'
        if i != count_x - 1:
            if in_split == 1:
                merge_temp = x_i.split(' ')
                x_i = merge_temp[1]
            x_ip = x_i + " " + x[i + 1]
            x_ip = x_ip.capitalize()
            if PGM_h.h("", "phrase", x_ip, 1, 5) == 0:
                fourth_oi = "phrase"
                fourth_yi = x_ip
                in_merge = 1
                in_phrase = 1
        x_i = fourth_yi
        if in_phrase == 1:
            fourth_yi = "\"" + fourth_yi + "\""
        best_y1.append(first_yi)
        best_y2.append(second_yi)
        best_y3.append(third_yi)
        best_y.append(fourth_yi)
        bfirst_oi.append(first_oi)
        bsecond_oi.append(second_oi)
        bthird_oi.append(third_oi)
        bfourth_oi.append(fourth_oi)
        grades.append(str(temp[4]) + ' ' + str(temp[5]) + ' ' + str(temp[6]))
        if in_split == 1:
            merge_temp = fourth_yi.split(' ')
            x_i = merge_temp[1]
        yi_pre = x_i
    print(x)
    print(bfirst_oi)
    print(best_y1)
    print(bsecond_oi)
    print(best_y2)
    print(bthird_oi)
    print(best_y3)
    print(bfourth_oi)
    tempo = ''
    for printy in best_y:
        tempo = tempo + ' ' + printy
    print(tempo)


def f_err_correction(o_list, yp, temp, landa_tmp):
    oy = temp
    oy0 = [oy[0], '']
    oy1 = [oy[1], '']
    bpr = [-99, -99]
    x = oy[1]
    splitcheck = oy[2]
    tempx = [x]
    if splitcheck == 1:
        tempx = x.split(' ')
    for x_counter in range(len(tempx)):
        inix = tempx[x_counter]
        if x_counter == 1:
            yp = oy1[0]
        for temp_oi in o_list:
            yi_list = make_list(temp_oi, inix, 1)
            for temp_yi in yi_list:
                temp_pr = pro_yox(inix, yp, temp_yi, temp_oi, landa_tmp, lang_model_, k_features, 0)
                if temp_pr > bpr[x_counter]:
                    oy0[x_counter] = temp_oi
                    oy1[x_counter] = temp_yi
                    bpr[x_counter] = temp_pr
    if splitcheck == 1:
        oy[0] = oy0[0] + ' ' + oy0[1]
        oy[1] = oy1[0] + ' ' + oy1[1]
        best_pr = (bpr[0] + bpr[1]) / 2
    else:
        oy[0] = oy0[0]
        oy[1] = oy1[0]
        best_pr = bpr[0]
    oy[4] = log(best_pr, 10)
    return oy


def f_split(temp, landa_tmp):
    best_pr = -99
    oy = temp
    oy[0] = 'nothing'
    x = oy[1]
    mergecheck = oy[3]
    if mergecheck == 0:
        yi_list = make_list("split", x, 1)
        for temp_yi in yi_list:
            temp_yi = temp_yi.replace("-", " ")
            if temp_yi in lang_model_.keys():
                temp_yi2 = temp_yi.replace(' ', '')
                temp_pr2 = -99
                if temp_yi2 in lang_model_1g.keys():
                    ix = lang_model_1g[temp_yi2]
                    temp_pr2 = float(ix)
                ix = lang_model_[temp_yi]
                temp_pr = float(ix)
                if temp_pr > temp_pr2 and temp_pr > best_pr:
                    oy[0] = "split"
                    oy[1] = temp_yi
                    best_pr = temp_pr / 2
                    oy[2] = 1
    oy[5] = best_pr
    return oy


def f_merge(xi1, temp, landa_tmp):
    best_pr = -99
    oy = temp
    oy[0] = 'nothing'
    x = oy[1]
    splitcheck = oy[2]
    if splitcheck == 0:
        if xi1 != '<\s>':
            temp_yi = x + xi1
            if temp_yi in lang_model_1g.keys():
                ix = lang_model_1g[temp_yi]
                temp_pr = float(ix)
                if temp_pr > best_pr:
                    oy[0] = "merge"
                    oy[1] = temp_yi
                    oy[3] = 1
                    best_pr = temp_pr
    oy[6] = best_pr
    return oy


# ------------------------------------------------------------------------------------------
# landa_temp = optimize.minimize(pr_function, landa_init, method = 'L-BFGS-B'
# , bounds= ((2.01e-1, 5), (2.01e-5, 5), (2.01e-5, 5), (2.01e-1, 5), (2.01e-1, 5), (2.01e-15, 5)))
# landa_temp = optimize.minimize(pr_function, landa_init, method='L-BFGS-B')
# landa = landa_temp.x
landa = landa_init
# for counter in range(k_features):
#   landa[counter] += 0.5
# print(landa)
xt = "howare youdoing n abandoneds boko aree iou sured \"that\" your namen was correct califo rnia sanf rancisco chareg up agree with"
print(test_model(xt, landa))
