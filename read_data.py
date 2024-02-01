import os
import xml.etree.ElementTree as ET
import re
import h5py
import numpy as np

def get_txt_files(folder_path):

    # 遍历指定文件夹
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # 检查文件是否为.txt文件
            #print(file)
            if file.endswith(".txt"):
                # 将文件的完整路径加入列表
                txt_files.append(os.path.join(root, file))
    return txt_files # 返回生成的列表

def check_type_attribute(xml_file, mahjong_type=169): #检查type属性是否为169
    # 解析XML文件
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # 遍历XML文件中的所有元素
    for elem in root:
        if elem.tag == 'GO':
            # 检查type属性是否为169
            return elem.get('type') == str(mahjong_type) or elem.get('type') == '225'
        else:
            pass

    return False
    

def check_type(file_path, mahjong_type=169):
        for file_path_ in file_path:
            if not check_type_attribute(file_path_, mahjong_type):
                file_path.remove(file_path_)


def mahjong_iterator(file_path, mahjong_type=169):
    #check_type(file_path, mahjong_type)
    #print(file_path)
    for file_path_ in file_path:
        mahjong_xml.parse_mahjong_log(file_path_)
    #return file_path

def mahjong_iterator_test(file_path, mahjong_type=169):
    file_num = 0
    line_num = 0
    #check_type(file_path, mahjong_type)
    for file_path_ in file_path:
        print(file_path_)
        if check_type_attribute(file_path_, mahjong_type):
            mahjong_iterator_ = mahjong_xml(file_path_,file_num,line_num)
            file_num,line_num = mahjong_iterator_.parse_mahjong_log(file_path_)
        else:
            pass
        #mahjong_iterator_.test_class()
    #return file_path
        
def mahjong_represent():
    hai = ["一", "二", "三", "四", "五", "六", "七", "八", "九", #萬子
       "①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", #筒子
       "1", "2", "3", "4", "5", "6", "7", "8", "9", #索子
      "東", "南", "西", "北", "白", "發", "中"]

class mahjong_xml(object):
    def __init__(self, file_path,file_num,line_num, if_write = True):
        self.file_path = file_path
        self.round,self.this_round,self.dealer,self.oya = 0,0,0,0 #局顺，本场，供托, 庄家
        self.DORA = [] #宝牌
        self.hai = []
        self.hai0 = [] #手牌1
        self.hai1 = [] #手牌2
        self.hai2 = [] #手牌3
        self.hai3 = [] #手牌4
        self.hai_discard = [[],[],[],[]] #弃牌

        self.hai_meld = [[],[],[],[]] #副露 

        self.score = [250,250,250,250] #分数


        self.if_write = if_write  #是否写入文件,不写入将返回记录
        self.file_count = file_num
        self.line_count = line_num
        self.line_count_max = 10000 

        #self.tree = ET.parse(file_path_)
    
    def parse_mahjong_log(self,file_path):
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        for elem in root.iter():
            #print(self.hai_discard)
            #print(elem.tag)
            if elem.tag == 'INIT':
                self.process_init(elem) #初始化牌局

            elif re.match(r'[A-Z]\d{1,3}', elem.tag):
                #print(elem.tag)
                self.process_process(elem.tag) #牌局过程

            elif elem.tag == 'AGARI':
                self.process_agari(elem) #和牌
                pass

            elif elem.tag == 'RYUUKYOKU':
                pass

            elif elem.tag == 'N':
                self.process_n(elem) #鸣牌
                pass

            elif elem.tag == 'DORA':
                self.DORA.append(int(elem.get('hai')))

            else:
                pass
        
        return self.file_count,self.line_count


    def process_init(self,elem):
        seed = elem.get('seed').split(',')
        self.round = seed[0]
        self.this_round = seed[1]
        self.dealer = seed[2]
        self.DORA = []
        self.DORA.append(int(seed[5]))
        ten = list(map(int,elem.get('ten').split(',')))
        self.score[0] = ten[0]
        self.score[1] = ten[1]
        self.score[2] = ten[2]
        self.score[3] = ten[3]
        self.oya = elem.get('oya')
        self.hai = []
        self.hai_discard = [[],[],[],[]]
        self.hai_meld = [[],[],[],[]]
        self.hai,self.hai0, self.hai1, self.hai2, self.hai3 = [],[],[],[],[]
        self.hai0.extend(list(map(int, elem.get('hai0').split(',')))) #手牌1
        self.hai1.extend(list(map(int, elem.get('hai1').split(',')))) #手牌2
        self.hai2.extend(list(map(int, elem.get('hai2').split(',')))) #手牌3
        self.hai3.extend(list(map(int, elem.get('hai3').split(',')))) #手牌4

        self.hai0.sort()
        self.hai1.sort()
        self.hai2.sort()
        self.hai3.sort()

        self.hai.append(self.hai0)
        self.hai.append(self.hai1)
        self.hai.append(self.hai2)
        self.hai.append(self.hai3)

        #print(self.hai)
        #print(self.hai[0])
        #print(self.hai0)
        #print(seed)

    def process_process(self,elem):
        match = re.match(r'([A-Z])(\d{1,3})', elem)
        letter = match.group(1)
        num = int(match.group(2))
        if letter == 'T':
            self.hai[0].append(num)
        elif letter == 'U':
            self.hai[1].append(num)
        elif letter == 'V':
            self.hai[2].append(num)
        elif letter == 'W':
            self.hai[3].append(num)
        elif letter == 'D':
            #print(self.hai[0])
            #print(num)
            self.hai[0].remove(num)
            self.hai_discard[0].append(num)
            self.record_discard(num,0)
        elif letter == 'E':
            self.hai[1].remove(num)
            self.hai_discard[1].append(num)
            self.record_discard(num,1)
        elif letter == 'F':
            self.hai[2].remove(num)
            self.hai_discard[2].append(num)
            self.record_discard(num,2)
        elif letter == 'G':
            self.hai[3].remove(num)
            self.hai_discard[3].append(num)
            self.record_discard(num,3)

    def process_n(self,elem):
        who = int(elem.get('who'))

        type = bin(int(elem.get('m')))[2:]
        type = type.zfill(16)


        #print(type)
        if type[-3] == '1': #吃
            meld_who = (int(type[14:],2)+who)%4
            be_chi = self.hai_discard[meld_who][-1]
            posit = be_chi//4
            self.hai_discard[meld_who].pop(-1)
            meld_bin = int(type[0:6],2)
            place = meld_bin%3 #余数表示被鸣的牌是哪一枚，1表示是中间大小的那一枚（0表示数字最小，2表示最大）
            #kind = meld_bin//3
            #kind_ = kind//7
            #kind__ = kind%7

            small_mod4 = int(type[11:13],2)
            big_mod4 = int(type[7:9],2)
            medium_mod4 = int(type[9:11],2)
            if place == 0:
                """
                chi_list_1 = [4*(posit+1),4*(posit+1)+1,4*(posit+1)+2,4*(posit+1)+3]
                chi_list_2 = [4*(posit+2),4*(posit+2)+1,4*(posit+2)+2,4*(posit+2)+3]
                set_1 = set(chi_list_1)
                set_2 = set(chi_list_2)
                set_ = set(self.hai[who])
                chi_1 = set_1.intersection(set_).pop()
                chi_2 = set_2.intersection(set_).pop()
                self.hai[who].remove(chi_1)
                self.hai[who].remove(chi_2)
                self.hai_meld[who].append([be_chi,chi_1,chi_2]) #写得很丑陋
                """
                chi_1 = 4*posit+small_mod4
                chi_2 = 4*(posit+1)+medium_mod4
                chi_3 = 4*(posit+2)+big_mod4
                self.hai[who].remove(chi_2)
                self.hai[who].remove(chi_3)
                self.hai_meld[who].append([be_chi,chi_2,chi_3])

            elif place == 1:
                chi_1 = 4*(posit-1)+small_mod4
                chi_2 = 4*posit+medium_mod4
                chi_3 = 4*(posit+1)+big_mod4
                self.hai[who].remove(chi_1)
                self.hai[who].remove(chi_3)
                self.hai_meld[who].append([be_chi,chi_1,chi_3])

            elif place == 2:
                chi_1 = 4*(posit-2)+small_mod4
                chi_2 = 4*(posit-1)+medium_mod4
                chi_3 = 4*posit+big_mod4
                self.hai[who].remove(chi_1)
                self.hai[who].remove(chi_2)
                self.hai_meld[who].append([be_chi,chi_1,chi_2])

            #print(meld_bin)
        elif type[-4] == '1': #碰
            meld_who = (int(type[14:],2)+who)%4
            meld = int(type[0:7],2)
            meld_type = meld//3
            pong_list = [4*meld_type,4*meld_type+1,4*meld_type+2,4*meld_type+3]
            pong_not = int(type[9:11],2)
            pong_list.remove(4*meld_type+pong_not)
            self.hai_discard[meld_who].pop(-1)
            self.hai_meld[who].append(pong_list)


        elif type[-5] == '1': #加杠
            meld_who = (int(type[14:],2)+who)%4
            meld = int(type[0:7],2)
            meld_type = meld//3
            kang_plus_ = int(type[9:11],2)
            kang_plus = 4*meld_type+kang_plus_
            self.hai_meld[who].append([kang_plus])
            self.hai[who].remove(kang_plus)
        elif type[9] == '1': #拔北 目前只研究四麻，不拔北
            #meld = type[10:13]
            pass
        else: #暗杠/明杠
            meld_who = (int(type[14:],2)+who)%4
            if meld_who == who:  #暗杠
                #kang = self.hai[who][-1]
                kang_ = int(type[0:8],2)
                meld_type = kang_//4
                kang_ = [4*meld_type,4*meld_type+1,4*meld_type+2,4*meld_type+3]
                self.hai_meld[who].append(kang_)
                self.hai[who].remove(4*meld_type)   
                self.hai[who].remove(4*meld_type+1)
                self.hai[who].remove(4*meld_type+2)
                self.hai[who].remove(4*meld_type+3) 
            else:
                kang = self.hai_discard[meld_who][-1]
                meld_type = kang//4
                #print("kang",kang)
                kang_ = [4*meld_type,4*meld_type+1,4*meld_type+2,4*meld_type+3]
                #print("kang_",kang_)
                self.hai_meld[who].append(kang_)
                self.hai_discard[meld_who].pop(-1)
                kang_.remove(kang)
                #print("kang__",kang_)
                self.hai[who].remove(kang_[0])
                self.hai[who].remove(kang_[1])
                self.hai[who].remove(kang_[2])

            #pass
                #明杠/暗杠

    def process_agari(self,elem):
        who = elem.get('who')
        fromWho = elem.get('fromWho')
        ten = elem.get('ten')
        yaku = elem.get('yaku')
        #yaku_name = elem.get('yaku_name')
        m = elem.get('m')
        sc = elem.get('sc')
        #self.record_agari(who,ten,yaku,m)
        
    def simp_1(self, record_):  #一种处理数据的函数，以后改数据就用这个模板，record_到record
        #print('record______',record_)
        hai_discard_copy = record_['hai_discard'].copy()
        hai_meld_copy = record_['hai_meld'].copy()
        score_copy = record_['score'].copy()
        
        hai_own = record_['hai'][record_['own']]
        discard_own = hai_discard_copy[record_['own']]
        #print(discard_own)
        hai_discard_copy.remove(discard_own)
        #print(record_['hai_discard'])
        meld_own = hai_meld_copy[record_['own']]
        hai_meld_copy.remove(meld_own)   #目前是把场风自风这些东西先舍弃掉，后面再考虑怎么加入这些因素
        score_own = score_copy[record_['own']]
        score_copy.remove(score_own)
        record = {
            'discard': record_['discard'], #自家舍牌 label
            'round': record_['round'],   #局顺  2
            'oya': record_['oya'],       #庄位  3
            'DORA': record_['DORA'],     #宝牌  4
            'hai_own': hai_own,          #自家手牌  5
            'discard_own': discard_own,  #自家牌河  6
            'discard_else': hai_discard_copy, #三家牌河  7
            'meld_own': meld_own,        #自家副露  8
            'meld': hai_meld_copy,       #三家副露  9
            'score_own': score_own,      #自家得分  10
            'score': score_copy          #三家得分  11
        }
        #print('record',record)
        self.print_data(record)


    def record_discard(self,num ,own):
        hai,hai_discard,hai_meld,score = self.hai, self.hai_discard, self.hai_meld, self.score
        record_ = {
            'own':own,                           #自家位置
            'discard':num,                       #弃牌
            'round':self.round,                  #局顺
            'this_round':self.this_round,        #本场
            'dealer':self.dealer,                #供托
            'oya':self.oya,                      #庄位
            'DORA':self.DORA,                    #宝牌
            'hai':hai,                      #手牌
            'hai_discard':hai_discard,      #弃牌
            'hai_meld':hai_meld,            #副露
            'score':score                   #分数
        }
        record_['hai_meld'] = flatten_innermost(record_['hai_meld'])
        #可以更换为其他的函数来改变打印的输出
        self.simp_1(record_)
        
        

        
        
    def print_data(self, record):
        if self.if_write:       
        #print(record)
            if self.line_count < self.line_count_max :
                file_name = 'data/discard/discard_{}.txt'.format(self.file_count)
                with open(file_name, 'a', encoding='utf-8') as file:
                    file.write(f'{self.line_count}$')
                    for key,value in record.items():
                        file.write(f"{value}$")
                
                    file.write('\n')
                self.line_count = self.line_count + 1
                #print(self.line_count)
            else:
                self.file_count = self.file_count + 1
                self.line_count = 0
                file_name = 'data/discard/discard_{}.txt'.format(self.file_count)
                with open(file_name, 'w', encoding='utf-8') as file:
                    file.write(f'{self.line_count}$')
                    for key,value in record.items():
                        file.write(f"{value}$")
                
                    file.write('\n')
                self.line_count = self.line_count + 1
        else: 
            record_ = flatten_record(record)
            
            #写为hdf5文件
            
            filename = 'hdf/discard.h5'

            with h5py.File(filename, 'a') as hdf_file:                
                group_name = f'data_{self.line_count}'
                group = hdf_file.create_group(group_name)
            # 遍历字典中的每个键值对
                for key, value in record_.items():
                    if isinstance(value, int):
                # 如果值是整数，将其存储为标量数据集
                        group.create_dataset(key, data=value)
                    elif isinstance(value, list):
                # 如果值是列表，将其转换为NumPy数组，然后存储为数据集
                        value = np.array(value)
                        group.create_dataset(key, data=value)
            #return record
            self.line_count = self.line_count + 1
            

    def test_class(self):
        print(self.round,self.this_round,self.dealer,self.oya)
        print(self.DORA, "宝牌")
        print(self.hai0, "手牌1")
        print(self.hai1, "手牌2")
        print(self.hai2, "手牌3")
        print(self.hai3, "手牌4")
        print(self.score, "分数")
    
def flatten(nested_list):
        flat_list = []
        indices = []
        start = 0
        for sublist in nested_list:
            end = start + len(sublist)
            indices.append(end)
            flat_list.extend(sublist)
            start = end
        return flat_list, indices

def flatten_innermost(nested_list):
    result = []
    for sublist in nested_list:
        if sublist:  # 检查子列表是否非空
            # 扁平化最内层的列表
            flattened = [item for inner_list in sublist for item in inner_list]
            result.append(flattened)
        else:
            # 对于空列表，直接添加到结果中
            result.append(sublist)
    return result


def flatten_record(record):

    flat_hai, ind_hai = flatten(record['hai'])
    #print(flat_hai)
    flat_hai = flat_hai + [-1]*(52 - len(flat_hai))
    #print(flat_hai)
    flat_discard, ind_dis = flatten(record['hai_discard'])
    flat_discard = flat_discard + [-1]*(72 - len(flat_discard))
    meld = flatten_innermost(record['hai_meld'])
    flat_meld, ind_meld = flatten(meld)
    flat_meld = flat_meld + [-1]*(48 - len(flat_meld))
    record['DORA'] = record['DORA'] + [-1]*(9 - len(record['DORA']))

    fl_record = {
        'own':record['own'],
        'discard':record['discard'],
        'round':record['round'],
        'this_round':record['this_round'],
        'dealer':record['dealer'],                #供托
        'oya':record['oya'],                      #庄位
        'DORA':record['DORA'],                    #宝牌
        'hai':flat_hai,  
        'ind_hai':ind_hai,                 #手牌
        'hai_discard':flat_discard,     #弃牌
        'ind_dis':ind_dis,
        'hai_meld':flat_meld,           #副露
        'ind_meld':ind_meld,
        'score':record['score']                   #分数
    }
    #print(fl_record)
    return fl_record

            
# 使用示例
if __name__ == "__main__":
    folder_path = 'data/xml2017'  # 替换为您的文件夹路径
    #folder_path = 'data/test'
    txt_files = get_txt_files(folder_path)
    #print(txt_files)
    mahjong_iterator_test(txt_files)
