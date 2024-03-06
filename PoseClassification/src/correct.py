import math
import pandas as pd
import sys
sys.path.append('D://Taichi//pytorch-openpose')


class Correct():
    def __init__(self,candidate, subset):
        self.candidate = candidate
        self.subset = subset 
        self.teachTextfile = 'src/correct_text.csv'
        self.keypoints = self.Bone2keypoints(candidate,subset)

        
        
    def pointDistance(self):
        """
        计算坐标点之间的距离
        """
        keyPoint = self.keypoints
        distance0 = self.mydistance(keyPoint[9],keyPoint[12])
        distance1 = self.mydistance(keyPoint[4],keyPoint[7])
        distance2 = self.mydistance(keyPoint[3],keyPoint[6])
        

        return [distance0,distance1,distance2]

    def pointAngle(self):
        '''
        计算关键点之间的角度
        '''
        keyPoint = self.keypoints
        angle0 = self.myAngle(keyPoint[1], keyPoint[2], keyPoint[3])
        angle1 = self.myAngle(keyPoint[1], keyPoint[5], keyPoint[6])
        angle2 = self.myAngle(keyPoint[2], keyPoint[3], keyPoint[4])
        angle3 = self.myAngle(keyPoint[5], keyPoint[6], keyPoint[7])
        angle4 = self.myAngle(keyPoint[4], keyPoint[8], keyPoint[9])
        angle5 = self.myAngle(keyPoint[7], keyPoint[11], keyPoint[12])
        angle6 = self.myAngle(keyPoint[1], keyPoint[8], keyPoint[9])
        angle7 = self.myAngle(keyPoint[1], keyPoint[11], keyPoint[12])
        angle8 = self.myAngle(keyPoint[8], keyPoint[9], keyPoint[10])
        angle9 = self.myAngle(keyPoint[11], keyPoint[12], keyPoint[13])

        return [angle0, angle1, angle2, angle3, angle4, angle5, angle6, angle7,
                angle8, angle9]

    def myAngle(self, A, B, C):
        '''
        余弦公式
        '''
        if A and B and C:
            c = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
            a = math.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
            b = math.sqrt((A[0] - C[0]) ** 2 + (A[1] - C[1]) ** 2)
            if 2 * a * c != 0:
                return (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
        else:
            return None
    
    def mydistance(self,A,B):
        '''
        距离公式
        '''
        if A and B:
            return (A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2
        else:
            return -99


    def Bone2keypoints(self,candidate, subset):
        '''
        若某关节点数据丢失或检测不到，则为None
        '''
        candi_get=[]
        joint_id = []
        keypoints=[None]*18


        if len(subset):  # 关节点需要非空才能计算
            # 获取对应关节编号
            for i in range(subset.shape[1]-2):
                if subset[0][i] >= 0:
                    joint_id.append(i) 
            # 获取坐标点
            for candi in candidate:
                candi_get.append([candi[0],candi[1]])
            # 将关节点与坐标点对应
            for index in range(len(joint_id)):
                keypoints[joint_id[index]] = candi_get[index]
        return keypoints


    def actionCorrect(self,act_id,bias):
        introduce = []
        angleFile = open('images/standPoseAngle.txt')
        standPoseAngle = angleFile.readlines()

        pointAngle = self.pointAngle()

        thisActAngle = standPoseAngle[act_id]
        thisActAngle = thisActAngle.replace('[','')
        thisActAngle = thisActAngle.replace(']','')
        thisActAngle = thisActAngle.replace('\n','')
        thisActAngle = thisActAngle.split(',')
        print(thisActAngle)
        for index in range(len(pointAngle)):
            # 有角度则进行角度对比
            one_introduce = {}
            gap = abs(pointAngle[index] - float(thisActAngle[index]) + bias)
            one_introduce['id'] = index
            one_introduce['gap'] = gap
            
            if pointAngle[index] and \
                pointAngle[index] + bias < float(thisActAngle[index]) :
                # 余弦值比标准动作小，说明动作的角度太大
                one_introduce['intro'] = 0
                
            else:
                one_introduce['intro'] = 1

            introduce.append(one_introduce)
            # 倒序排列（gap从大到小）
            introduce.sort(key=lambda x:x['gap'],reverse=True)
        return introduce

    # 根据Introduce返回对应的指导内容
    def getIntroWord(self,act_id,bias):
        introduce_dict = self.actionCorrect(act_id,bias)
        rows = []  #记录要提取的行号
        intro = [] #记录建议，0或1
        intro_text = []
        # 读取csv文件
        text_data = pd.read_csv('src/correct_text.csv',encoding='utf-8')
        for i in introduce_dict[:3]:
            rows.append(i['id'])
            intro.append(i['intro'])
        
        # print(rows)
        # print(intro)
        for r in rows:
            if intro[0] < 1:
                intro_text.append(text_data.loc[r,'0']) 
            else:
                intro_text.append(text_data.loc[r,'1']) 

        return intro_text