import numpy as np 
import time
import threading
import os
if "DISPLAY" not in os.environ.keys():
    import matplotlib
    matplotlib.use('agg')
import matplotlib.pyplot as plt
FlagA = False
def Thread_template(File):
    global FlagA
    Reward = Generate_virtual_data()
    File([Reward,1],["Reward","HG"])
    Valid_Loop(3)
    FlagA = True

def Display(VM_moudle,File):
    global FlagA
    print("AAA")
    print(File)
    print (FlagA)
    FlagA=False
    
    while(not FlagA):
        print (not FlagA)
        VM_moudle(File)
    
# import 
# Enable_Flag = 0

def main():
    Access_Type_Record_file()
    File1 = Record_File()
    File2 = Record_File()
    VM = My_visualize_moudle("HIHI","curve","eps","acc(%)")
    VM2 = My_visualize_moudle("HAHA","curve","eps","acc(%)")
    for i in range(10):
        T1 = threading.Thread(target=Thread_template,args=(File1,))
        T2 = threading.Thread(target=Thread_template,args=(File2,))
        T3 = threading.Thread(target=Display,args=(VM,File1))
        T4 = threading.Thread(target=Display,args=(VM2,File2))
        T1.start()
        T2.start()
        T3.start()
        T4.start()
        # Valid_Loop(3)
        T1.join()
        FlagA = True
        T2.join()
        FlagB = True
        T3.join()
        T4.join()
        # Reward = Generate_virtual_data()
        # Plot moudle
        # File([Reward],["Reward"])
        # File1
        # print(File)
        # Enable_Flag = 1
        # VM(File1)
        # VM2(File2)
    pass
    
def Access_Type_Record_file():
    print(color.HEADER+"Demo Record file using"+color.ENDC)
    RF, RFB = Record_File(),Record_File()
    Data_name = ["Leo","Meta"]
    Data_name2 = ["Leo1","Meta1"]
    RF([20,5],Data_name)
    RF([7],["Leo"])
    RF([18],["Meta"])
    RFB([777,888],Data_name)
    RFB([72],["Leo"])
    RFB([17],["Meta"])
    RFB([2424,7878],Data_name2)
    print(color.GREEN+"RF : {}".format(RF)+color.ENDC)
    print(color.WARNING+"RFB : {}".format(RFB)+color.ENDC)
    S = RF+RFB
    print(color.BLUE+"Add together: {}".format(S)+color.ENDC)
def Valid_Loop(T):
    time.sleep(T)

    return
def Generate_virtual_data():
    # if Enable_Flag == 0:
    #     global() ["count_Generate_virtual_data"] = 0
    # else 
    #     count_Generate_virtual_data +=1
    Reward = np.random.normal(0,1,size=1)
    # if count_Generate_virtual_data % 10 == 0 && count_Generate_virtual_data !=0:
                

    
    # return Reward,
    return float(Reward)
class Record_File(object):
    def __init__(self):
        self.data_item = {}
        pass
    def Register_data(self,Type,Name):
        if Type == "List":
            self.data_item[Name] = []
        else:
            print("We don't have this type")
            raise
    def Access_data(self):
        return self.data_item.keys(),self.data_item
    def __call__(self,data,Name):
        if isinstance(data,list) ==False:
            raise
        if isinstance(Name,list) ==False:
            raise
        if len(data)!=len(Name):
            raise
        for i in range(len(Name)):
            if Name[i] in self.data_item.keys():
                self.data_item[Name[i]].append(data[i])
            else:
                self.data_item[Name[i]] = [data[i]]
                
    def __len__(self):
        return len(self.data_item.keys())
    def __add__(self, Other_Record):

        if isinstance(Other_Record,Record_File) == False:
            raise
        
        Data_out = self.data_item.copy()
        for name in Other_Record.data_item.keys():
            if name in self.data_item.keys():
                Data_out[name].extend(Other_Record.data_item[name])
            else:
                Data_out[name]=Other_Record.data_item[name]
        
        new_record_file = Record_File()
        Data_item_list = [Data_out[name] for name in Data_out.keys()]
        new_record_file(Data_item_list,Data_out.keys())

        return new_record_file
    def __iadd__(self,Other_Record):
        if isinstance(Other_Record,Record_File) == False:
            raise
        for name in Other_Record.data_item.keys():
            if name in self.data_item.keys():
                self.data_item[name].extend(Other_Record.data_item[name])
            else:
                self.data_item[name]=Other_Record.data_item[name]
        return self.data_item
    def __str__(self):
        Print_out = "".join(
            ["\n%s:\n"%name+"".join(
                ["{},\t".format(self.data_item[name][i]) for i in range(len(self.data_item[name]))]
                ) for name in self.data_item.keys()]
        )
        
        return Print_out
    def __format__(self,format):
        Print_out = "".join(
            ["\n%s:\n"%name+"".join(
                ["{},\t".format(self.data_item[name][i]) for i in range(len(self.data_item[name]))]
                ) for name in self.data_item.keys()]
        )
        
        return Print_out
    # def __iter__(self):
    #     return
# class My_Matplot():
#     def __init__(self):
#         pass
class My_visualize_moudle(object):
    def __init__(self,WindowName,FigureTitle,axis_X,axis_Y,save_png_path= "./Visualization/"):
        import seaborn as SB
        #import matplotlib
        #matplotlib.use('agg')
        #import matplotlib.pyplot as plt
        # plt.ion()
        self.figure = plt.figure(WindowName)
        
        self.Window_Name = WindowName
        self.Title = FigureTitle
        self.axis_X=axis_X
        self.axis_Y=axis_Y
        # self.figure.title(self.Title)
        self.window_show = self.figure.add_subplot(111)
        self.save_path = save_png_path
        # for i in range()
        #     SB.lineplot(x=)
        #     SB.lineplot(x=len())
        # self.figure.show()
        pass
    def __call__(self,data):
        #import matplotlib
        #matplotlib.use('agg')
        #import matplotlib.pyplot as plt
        import seaborn as SB
        #import matplotlib.finance as mpf
        SB.set(style='white',palette='muted',color_codes=True)
        # seaborn.plt.line2D
        # self.figure.cla()
        self.window_show.cla()
        self.window_show.set_title(self.Title)
        self.window_show.set_xlabel(self.axis_X)
        self.window_show.set_ylabel(self.axis_Y)
        # self.figure.
        # self.window_show.title(self.Title)
        Name, data = data.Access_data()
        color_i=0
        color_form = SB.color_palette()
        for name in Name:
            SB.lineplot(x=[i for i in range(len(data[name]))],y=data[name],ax=self.window_show,hue=[name for i in range(len(data[name]))],
            legend="full",markers=True,palette=[color_form[color_i]])
            color_i = color_i+1
            # mpf.candlestick_ohlc(XX,L,width=3,colorup='r',colordown='green')
        # plt.savefig("")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.figure.savefig(os.path.join(self.save_path,self.Window_Name+".png"),dpi=700)
        # self.figure.canvas.draw_idle()

        # plt.pause(3)
class color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
if __name__=="__main__":
    
    main()
