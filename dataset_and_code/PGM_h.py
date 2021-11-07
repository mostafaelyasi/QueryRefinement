__author__ = 'Mostafa Elyasi'

# input
#   yi = suggested word
#   oi = operations that could be 1-del 2-ins 3-subs 4-trans 5-merge 6-split 7-phrase 8-nothing
#   xi = original word
#   features = type of feature that we want to check
#       1- freq of Xi is more than freq of Yi then return 0
#       2- xi is in stop words return 1
#       3- single word query(1) or multi word query(0)
#       4- exact 3grams query of xi return 0
#       5- xi in phrase(0) else(1)
#       6- only check operations
#   length_of_check = an integer
#       the function will check the letters deletion, insertion and substitution with maximum length of length_of_check
# output
#   0- oi operation cannot convert xi to yi
#   1- oi operation can convert xi to yi
#
with open('1gram_train5k.txt', 'r') as f:  
    freq = {}
    l = f.read().split('\n')      
    for i in l:
        values = i.split('\t')   # Split using 'tab'
        freq[values[1]] = values[0]
    lang_model_1g = freq
    f.close()

with open('stopwords.txt', 'r') as f:  
    sw = f.read()     
    f.close()

with open('3grams.txt', 'r') as f:  
    three_grams = f.read()     
    f.close()

with open('phrase2280.txt', 'r') as f:  
    phrase = f.read()     
    f.close()


def h(yi, oi, xi, length_of_check, features):
    yi = yi.lower()
    xi = xi.lower()
    if features == 1:
        if yi in lang_model_1g.keys():
            if xi in lang_model_1g.keys():
                x_pr = lang_model_1g[xi]
                y_pr = lang_model_1g[yi]
                if x_pr > y_pr:
                    return 0
            return 1
    elif features == 2:
        if xi in sw:
            return 0
        return 1
    elif features == 3:
        if xi.find(" ") != -1:
            return 0
        else:
            return 1
    elif features == 4:
        if xi in three_grams:
            return 0
        else:
            return 1
    elif features == 5:
        if xi in phrase:
            return 0
        else:
            return 1
    elif oi == "nothing" and yi == xi:
        return 1
    # first check for spell correction 1-del 2-ins 3-subs 4-trans
    elif xi.find(" ") == -1 and yi.find(" ") == -1:
        if oi == "del":
            for k in range(len(xi)):
                for j in range(1, length_of_check + 1):
                    first_part = xi[:k]  #
                    second_part = xi[k + j:]  #
                    temp_string = first_part + second_part
                    if yi == temp_string:
                        return 1
        elif oi == "ins":
            for k in range(len(xi)):
                for j in range(1, length_of_check + 1):
                    first_part = yi[:k]  #
                    second_part = yi[k + j:]  #
                    temp_string = first_part + second_part
                    if xi == temp_string:
                        return 1
        elif oi == "subs":
            if len(xi) == len(yi) and yi != xi:
                for k in range(len(xi)):
                    for j in range(1, length_of_check + 1):
                        xi_first_part = xi[:k]
                        xi_second_part = xi[k + j:]
                        xi_tmp_string = xi_first_part + xi_second_part
                        yi_first_part = yi[:k]
                        yi_second_part = yi[k + j:]
                        yi_tmp_string = yi_first_part + yi_second_part
                        if xi_tmp_string == yi_tmp_string:
                            return 1
        elif oi == "trans":
            if len(xi) == len(yi) and xi != yi:
                for k in range(len(xi) - 1):
                    temp1 = xi[k]
                    xi_first_part = xi[:k]
                    for j in range(k + 1, len(xi)):
                        temp2 = xi[j]
                        xi_second_part = xi[k + 1:j]
                        xi_third_part = xi[j + 1:]
                        xi_tmp_string = xi_first_part + temp2 + xi_second_part + temp1 + xi_third_part
                        if xi_tmp_string == yi:
                            return 1
    # second check for split and merge
    else:
        xi_tmp = xi
        yi_tmp = yi
        if oi == "merge" and xi_tmp.find(" ") != -1:
            xi_tmp = xi_tmp.replace(" ", "")
            if xi_tmp == yi:
                return 1
        elif oi == "split" and yi_tmp.find(" ") != -1:
            yi_tmp = yi_tmp.replace(" ", "")
            if yi_tmp == xi:
                return 1
        elif oi == "phrase":
            if yi in phrase:
                return 1
    return 0
