from PIL import ImageColor,ImageDraw,ImageFont,Image,ImageTk
import collections
import tkinter as tk
from tkinter import ttk
import cv2
import time
import mss
import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
import tensorflow as tf
import label_map_util
from pynput.keyboard import Key, Controller
from datetime import datetime
import  csv

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=4,
                               display_str_list=(),
                               use_normalized_coordinates=True):
 
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    
    draw.line([(left, top), (left, bottom), (right, bottom),
                 (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
      # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin

def select_boxes_and_labels(
    boxes,classes,scores,category_index,categories,
    max_boxes_to_draw=20,min_score_thresh=.5):
   
  # Create a display string (and color) for every box location, group any boxes
  # that correspond to the same location.
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    disp_list=[]

    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            #if classes[i] 
            
            #check if class exists in category index keys 
            #check if class is to be used
            if classes[i] in category_index.keys() and int(categories[categories['id']==classes[i]]['value'])==1:
                class_name = category_index[classes[i]]['name']
                disp_list.append(classes[i])

                display_str = '{}: {}%'.format(str(class_name), int(100*scores[i]))

                box_to_display_str_map[box].append(display_str)

                box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

    return box_to_display_str_map,box_to_color_map,disp_list

def draw_boxes_and_labels(image,box_to_display,box_to_color,use_normalized_coordinates=True,line_thickness=4):
    
    for box, color in box_to_color.items():
        ymin, xmin, ymax, xmax = box
    
        draw_bounding_box_on_image(
        image,
        ymin,
        xmin,
        ymax,
        xmax,
        color=color,
        thickness=line_thickness,
        display_str_list=box_to_display[box],
        use_normalized_coordinates=use_normalized_coordinates)

        
class Censor:
    
    def get_var(self,var):
        val=self.data[self.data['name']==var].iloc[0]['value']
        dtype=self.data[self.data['name']==var].iloc[0]['type']
        if dtype=='boolean' or dtype=='int' :
            val=int(val)
            
        return val
        
    def __init__(self, data,categories):
        self.data=data
        self.categories=categories
        
        self.sct=mss.mss() 
        
        print('initializing...')
        self.useGpu=self.get_var('useGpu')
        self.classify=self.get_var('classify')
        self.display_box=self.get_var('display_box')
        self.display_fps=self.get_var('display_fps')
        
        self.top=0
        self.left=0
        self.width=self.get_var('width')
        self.height=self.get_var('height')
        
        self.rwidth=self.get_var('resize_width')
        self.rheight=self.get_var('resize_height')
        
        self.use_num_frames=self.get_var('use_num_frames')
        self.frame_seconds=self.get_var('frame_seconds')
        self.perc_frames=self.get_var('perc_frames')/100
        self.num_frames=self.get_var('num_frames')
        
        self.skip_detection=self.get_var('skip_detection')
        self.log_detection=self.get_var('log_detection')
        self.log_stats=self.get_var('log_stats')
        self.image_log_path=self.get_var('image_log_path')
        self.ann_log_path=self.get_var('ann_log_path')
        self.stats_log_path=self.get_var('stats_log_path')
        
        self.class_counter= collections.defaultdict(list)
        #self.class_counter={cl:[0] for cl in categories['id'].tolist()} ###change classes to det
        for cl in categories['id'].tolist():
            self.class_counter[cl].append(time.time())
            
        self.max_boxes=self.get_var('max_boxes')
        self.glob_thresh=self.get_var('global_threshold')/100
        
        self.transparent=self.get_var('transparent')
        
        self.frz_graph=self.get_var('model_location')
        self.labels=self.get_var('pbtxt_file')
        
        self.new_img=Image.new('RGBA', (self.width, self.height), (255, 0, 0, 0))
        self.mon = {"top": self.top, "left": self.left, "width": self.width, "height": self.height}
        self.skipped=False
        self.fps=1
        self.det_fps=1
        self.avg_fps=1
        self.sum_fps=[1]
        
        self.last_time=time.time()

        self.tf_init()
        print('Running...')
        
    def read_fps(self):
            #fps = 0
        
        
        if (time.time() - self.last_time) < 1:
            self.fps+=1
        else:
            if self.skipped:
                self.skipped=False
            else:
                self.sum_fps.append(self.fps)
                self.sum_fps[:]=self.sum_fps[-15:]
                self.avg_fps= round(sum(self.sum_fps) /len(self.sum_fps))
                if self.display_fps:
                    print('avg fps',self.avg_fps,'curr fps',self.fps)
                
            self.last_time = time.time() 
            self.fps=0

    def get_frame(self):
        
        if self.classify:
            return  self.run_inference()
        else:
            
            return self.sct.grab(self.mon),False
    
    def run_detection(self,image):
       
        output_dict = self.session.run(self.tensor_dict,feed_dict={self.image_tensor: np.expand_dims(image, 0)})
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        return output_dict
    
    def df_counter(self,disp_list): 
        action=False
        class_id=None
        
        seconds=self.frame_seconds
        
        if self.use_num_frames:
            num_frames=self.num_frames
        else:
            perc_fps=round(self.perc_frames*(self.sum_fps[-1]*seconds))
            num_frames=perc_fps if perc_fps>1 else 1
            
        self.det_fps=num_frames
        
        array_cap=(self.avg_fps*(seconds+3))
        
        for cl in self.categories['id'].tolist():
            if cl in disp_list:
                self.class_counter[cl].append(time.time())
                self.class_counter[cl][:]=self.class_counter[cl][-array_cap:]
                
                now=time.time()    
                total=sum([1 if (now-count)<seconds else 0 for count in self.class_counter[cl]])
                #print('total {}, num_frames {}'.format(total,num_frames))
                if total>num_frames:
                    action=True
                    class_id=cl
                
        return action,class_id
            
    def run_inference(self):
        action=False
        orig=self.sct.grab(self.mon)
        image = np.asarray(orig, dtype=np.uint8)
       
        
        #image_resized=image
        #best quality for shrinking :
        #cv2.INTER_AREA
        image_resized=cv2.resize(image, (self.rwidth,self.rheight), interpolation=cv2.INTER_LINEAR)
        image_resized=cv2.cvtColor(image_resized[:,:,:3], cv2.COLOR_BGR2RGB)

        output_dict=self.run_detection(image_resized)
        
        box_to_display,box_to_color,disp_list=select_boxes_and_labels(output_dict['detection_boxes'][0],
                                                            output_dict['detection_classes'][0].astype(np.uint8),
                                                            output_dict['detection_scores'][0],
                                                            self.category_index,self.categories,
                                                            max_boxes_to_draw=self.max_boxes,
                                                            min_score_thresh=self.glob_thresh)
        
        
        if disp_list:
            to_act,class_id=self.df_counter(disp_list)
            
            if self.log_stats:
                self.record_stats(disp_list,to_act)
            
            if self.skip_detection:
                action=to_act

            if self.log_detection:
                if to_act:
                    self.record_detection(orig,box_to_display,box_to_color)
            if self.display_box:
                if self.transparent:
                    image=self.new_img.copy()
                else:
                    None
                
                draw_boxes_and_labels(image,box_to_display,box_to_color,use_normalized_coordinates=True,line_thickness=6)
                #image=Image.fromarray(image[:,:,[2,1,0]])
                #image=Image.fromarray(image[:, :, [2,1,0]])
                #image=Image.fromarray(image)
        else:
            image=self.new_img.copy()
            
            
        return image,action
    
    
        
    def record_stats(self,disp_list,acted):    
        ann_list=[]
        if not os.path.exists(self.image_log_path):
            os.mkdir(self.image_log_path)
            
        if os.path.exists(self.stats_log_path):
            mode='a'
        else:
            mode='w'
            
        with open(self.stats_log_path,mode,newline='') as f:
            
            wr = csv.writer(f,delimiter=",")
            if mode=='w':
                wr.writerow(['class_id','avg_fps','curr_fps','det_fps','acted'])
            for cl in disp_list:
                wr.writerow([cl,self.avg_fps,self.sum_fps[-1],self.det_fps,acted])

                
                
    def record_detection(self,image,classes,boxes):
        
        curr_date=datetime.now().strftime('%Y%m%d_%H%M%S%f')
        file_name='{}.png'.format(curr_date)
        
        if not os.path.exists(self.image_log_path):
            os.mkdir(self.image_log_path)
        
        if os.path.exists(self.ann_log_path):
            mode='a'
        else:
            mode='w'
        
        with open(self.ann_log_path,mode,newline='') as f:
            wr = csv.writer(f,delimiter=",")
            if mode=='w':
                wr.writerow(['filename','width','height','class','xmin','ymin','xmax','ymax'])
                
            for box,_ in boxes.items():
                ymin, xmin, ymax, xmax = box

                (xmin, xmax, ymin, ymax)=(xmin * self.width, xmax * self.width,ymin * self.height, ymax * self.height)
                wr.writerow([file_name,self.width,self.height,classes[box][0],xmin,ymin,xmax,ymax])

        mss.tools.to_png(image.rgb, (self.width,self.height), output=self.image_log_path+file_name)
        
        
    def tf_init(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.frz_graph, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
                
        self.category_index = label_map_util.create_category_index_from_labelmap(self.labels, use_display_name=True)
        
        if not self.useGpu:
            config = tf.ConfigProto(device_count = {'GPU': 0})
            self.session=tf.Session(graph=detection_graph,config=config)
        else:
            self.session=tf.Session(graph=detection_graph)

        all_tensor_names = {output.name for op in detection_graph.get_operations() for output in op.outputs}
        tensor_dict = {}
        for key in [ 'num_detections', 'detection_boxes', 'detection_scores','detection_classes' ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = detection_graph.get_tensor_by_name(tensor_name)
       
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.tensor_dict=tensor_dict
    
class App:
    
    def __init__(self, window):
        self.window = window
        self.window.title('SkipThis')
        
        self.data,self.categories=self.read_data()
        
        self.window.overrideredirect(self.get_var('hide_window'))        
        
        self.window.wm_attributes("-topmost", self.get_var('top_most'))
        self.transparent=self.get_var('transparent')
        
        if self.transparent:
            self.window.wm_attributes("-transparentcolor", "white")
        
        self.canvas = tk.Label(window, bg='white')
        self.canvas.pack(side="bottom", fill="both", expand="yes")

        self.vid = Censor(self.data,self.categories)
        self.det_skip_delay=self.get_var('det_skip_delay')
        self.keyboard_key=self.get_var('simulated_button')
       # self.last_time=time.time()
        self.update()
        
    def get_var(self,var):
        val=self.data[self.data['name']==var].iloc[0]['value']
        dtype=self.data[self.data['name']==var].iloc[0]['type']
        if dtype=='boolean' or dtype=='int' :
            val=int(val)
        return val
    
    def skip_action(self):
        keyboard = Controller()
        atr=getattr(Key, self.keyboard_key)
        keyboard.press(atr)
        keyboard.release(atr)
        #print('acted')
    
    def read_data(self):
        data = pd.read_csv('settings.csv')
        categories=pd.read_csv('categories.csv')     
        return data,categories
    
    def close(self):
        self.window.destroy()
    
    def update(self):
         # Get a frame from the video source
        
        frame,skip=self.vid.get_frame()
        
        if self.vid.display_box:
            
            photo = ImageTk.PhotoImage(image=frame)
            self.canvas.configure(image=photo)
            self.canvas.image = photo

        self.vid.read_fps()
        
        #add a longer delay if skip was triggered
        delay=1
        if skip:
            self.vid.skipped=True
            self.skip_action()
            delay+=self.det_skip_delay
        
        self.window.after(delay, self.update)

class Toolbar:
    #read settings
    def read_data(self):
        data = pd.read_csv('settings.csv')

        if os.path.isfile('categories.csv'):
            categories=pd.read_csv('categories.csv')     
        else:
            category_map=data[data['name']=='pbtxt_file'].iloc[0]['value']
            categories=self.read_category_map(category_map)

        return data,categories
    
    
    def read_category_map(self,pbtxt_file):
        label_dict=label_map_util.create_category_index_from_labelmap(pbtxt_file, use_display_name=True)
        labels=pd.DataFrame.from_dict(label_dict,orient='index')
        labels['value']='1'
        labels['type']='boolean'
        labels['label_map']=pbtxt_file
        labels.to_csv('categories.csv',index=False)
        return labels
    
    def get_tab(self,row):
        return self.tab_dict[row['tab']]
    
    def add_row(self,row_data,category=False):
        #Select Tab
        if not category:
            tab=self.get_tab(row_data)
            tab_no=row_data['tab']
            label_col='display'
            text=row_data[label_col]
            if not pd.isnull(row_data['units']):
                text+="({})".format(row_data['units'])
        else:
            tab=self.tab_dict[1]
            tab_no=1
            label_col='name'
            text=row_data[label_col]
        #Print Label
        tk.Label(tab, text=text).grid(row=2+self.tab_row[tab_no], column=0)

        #Print Variable
        var=self.populate_tk(row_data,tab,self.tab_row[tab_no])

        #increase the count for the tab so next iteration its +1 on the row
        self.tab_row[tab_no]+=1
        return var
    
    
    def update_settings(self):
        self.data['value']=[e.get() for e in self.data_vars]
        self.data.to_csv('settings.csv',index=False)

        self.categories['value']=[e.get() for e in self.cat_vars]
        self.categories.to_csv('categories.csv',index=False)
        category_map=self.data[self.data['name']=='pbtxt_file'].iloc[0]['value']
        label_map=self.categories.iloc[0]['label_map']
        if category_map!=label_map:
            self.read_category_map(category_map)
            
            
            
    def populate_tk(self,row,window,val,col='value'):
     
        if row['type']=='boolean':
            var=tk.IntVar()
            if int(row[col])==1:
                var.set(1)

            checkbox=tk.Checkbutton(window,variable=var)
            checkbox.grid(row=2+val, column=6)
        else:
            var=tk.Entry(window)
            var.insert(0,row[col])
            var.grid(row=2+val, column=6)
        
        return var
    
    def open_window(self):
        self.update_settings()
        self.app=App(tk.Toplevel())
        
        #button text and command replace 
        self.b.configure(text = "Stop", command=self.close_window)
        
    def close_window(self):
        self.app.close()
        
        self.b.configure(text = "Run", command=self.open_window)
        
    
    
    def __init__(self, window):
        
        self.window = window
        window.wm_title("SkipThis")
        
        #load data
        self.data,self.categories=self.read_data()

        self.tab_control = ttk.Notebook(window)
        
        self.tab_dict={} 
        for i in range(4):
            self.tab_dict[i] =  ttk.Frame(self.tab_control) 
        #display(self.tab_dict)
        
        self.tab_control.add(self.tab_dict[0], text='App Settings') 
        self.tab_control.add(self.tab_dict[1], text='Category Settings') 
        self.tab_control.add(self.tab_dict[2], text='Action Settings')
        self.tab_control.add(self.tab_dict[3], text='Model Settings')
        
        
        self.tab_row=[0]*4
        self.data_vars=[]
        self.cat_vars=[]
        
        for i in range(len(self.data)):
            #display(i,len(self.data))
            self.data_vars.append(self.add_row(self.data.iloc[i]))

        for i in range(len(self.categories)):
            self.cat_vars.append(self.add_row(self.categories.iloc[i],True))


        self.tab_control.pack(expand=1, fill='both')

        height=1
        width=6
        
        a=tk.Button(window, text="Update", command=self.update_settings, height = height, width = width)
        a.pack(padx=5, pady=5, side=tk.RIGHT)
        self.b=tk.Button(window, text="Run", height = height, width = width,command=self.open_window)
        self.b.pack(padx=5, pady=5, side=tk.RIGHT)
        
if __name__ == "__main__":
    root=tk.Tk()
    #root = tk.Toplevel()
    Toolbar(root) 
    root.mainloop()