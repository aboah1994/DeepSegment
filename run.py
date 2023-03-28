import torch
from torchvision import transforms

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
import tkinter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2, os, gdown
import numpy as np
import pandas as pd 
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def load_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kpt_file = 'yolov7-w6-pose.pt'
    if not os.path.isfile(kpt_file):
        url = 'https://drive.google.com/uc?id=1DLu3LnMmkvQQZ3T_d1k0aoZZ08x8lHvM'
        gdown.download(url,quiet=False)
    model = torch.load('yolov7-w6-pose.pt', map_location=device)['model']
    # Put in inference mode
    model.float().eval()

    if torch.cuda.is_available():
        model.half().to(device)
    return model

def run_inference(url):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image = cv2.imread(url) # shape: (480, 640, 3)
    # Resize and pad image
    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    output, _ = model(image.half().to(device)) # torch.Size([1, 45900, 57])
    return output, image

def run_vid_inference(model,image):

    image = letterbox(image, 960, stride=64, auto=True)[0] # shape: (768, 960, 3)
    # Apply transforms
    image = transforms.ToTensor()(image) # torch.Size([3, 768, 960])
    # Turn image into batch
    image = image.unsqueeze(0) # torch.Size([1, 3, 768, 960])
    output, _ = model(image.half().to(device)) # torch.Size([1, 45900, 57])
    return output, image
def visualize_output(output, image):
    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    
    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
    # plt.figure(figsize=(12, 12))
    # plt.axis('off')
    # plt.imshow(nimg)
    # plt.show()
    cv2.imshow('figure',nimg)
    cv2.waitKey(0)


def plot_face_kpts(im, kpts, steps, plot_kpts, orig_shape=None):
    #Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    kpt_out = []
    kid_out = []
    for kid in range(num_kpts):
        if kid < 5:
            r, g, b = pose_kpt_color[kid]
            # print ('keyp', kid, r,g,b)
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                # print (kid)
                # print ('kid printed')
                
                if kid == 3:
                    radius = 30
                elif kid == 4:
                    radius = 30
                # elif kid == 2: 
                #     radius=30
                else:
                    radius = 5
                if configs['plot_kpts']:
                    cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                # print ((int(x_coord), int(y_coord)))
                kpt_out.append([int(x_coord), int(y_coord)])
                kid_out.append(kid)
        else:
            r, g, b = pose_kpt_color[kid]
            # print (kid)
            radius = 2
            # if kid == 8: radius=30
            
            if kid == 10: radius=30 # wrist
            # if kid == 5: radius=30 ## left sholder
            # if kid == 6: radius=30 ## right shoulder
            if kid == 9: radius=30
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        continue
                if configs['plot_kpts']:
                    cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
                # print ((int(x_coord), int(y_coord)))
                kpt_out.append([int(x_coord), int(y_coord)])
                kid_out.append(kid)

    
    return kpt_out, kid_out

def vis_vid_output(output, image,model, plot_kpts):
    output = non_max_suppression_kpt(output, 
                                     0.25, # Confidence Threshold
                                     0.65, # IoU Threshold
                                     nc=model.yaml['nc'], # Number of Classes
                                     nkpt=model.yaml['nkpt'], # Number of Keypoints
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    # print (len(output[0]))
    nimg = image[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    # nimg = np.zeros((nimg.shape))
    kpts = {}; kids = {}
    for idx in range(output.shape[0]):
        # plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        kpt_out, kid_out = plot_face_kpts(nimg,output[idx, 7:].T, 3, configs['plot_kpts'])
        kpts[idx] = kpt_out
        kids[idx] = kid_out

        # print ('flames')

    return nimg, kpts, kids

def angle_calc(kpts, kids):
    data = kpts[0]
    kid_data = kids[0]
    data_id = {}
    for idd,dd in zip(kid_data,data):
        data_id[idd] = dd
    
    # print (data_id)

    angles = []
    angle_obj = {}
    # print (kpts)
    # print (kids)
    id_subs = [[4,2,0],[2,0,1],[0,1,3]]
    for id_sub in id_subs:
        if set(id_sub) <= set(kids[0]):
            cdata = [data_id[id_sub[0]],data_id[id_sub[1]],data_id[id_sub[2]]]
            if len(cdata) > 2:
                a = np.array(cdata[0])
                b = np.array(cdata[1])
                c = np.array(cdata[2])
                # print (a,b,c)

                ba = a - b
                bc = c - b

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                angles.append(np.degrees(angle))
                angle_obj[np.degrees(angle)] = id_sub

    return angles, angle_obj


def norm_xy(nimg,kpts,kids):
    # print (kids)
    img_width = nimg.shape[0]
    img_height = nimg.shape[1]
    x,y=zip(*kpts[0])
    xs = [round(i/img_width,3) for i in x]
    ys = [round(i/img_height,3) for i in y]

    ndata = {}
    for ix,iy,cid in zip(xs,ys,kids[0]):
        ndata[cid] = [ix,iy]
    return x,y,ndata

def calc_dev(x,y,ndata, kpts, kids):
    data = kpts[0]
    kid_data = kids[0]
    data_id = {}
    for idd,dd in zip(kid_data,data):
        data_id[idd] = dd
    
    # print (data_id)
    # print (x,y)

    angles = []
    angle_obj = {}
    angles_ = []
    angle_obj_ = {}
    id_subs = [[4,2,0],[2,0,1],[0,1,3]]
    # print (kids[0])
    # print ('------------')
    # print (data_id)
    # print (ndata)
    # print ('------------')
    for id_sub in id_subs:
        if set(id_sub) <= set(kids[0]):
            
            cdata = [data_id[id_sub[0]],data_id[id_sub[1]],data_id[id_sub[2]]]
            ndata_ = [ndata[id_sub[0]],ndata[id_sub[1]],ndata[id_sub[2]]]
            
            
            if len(ndata_) > 2:
                a = np.array(ndata_[0])
                b = np.array(ndata_[1])
                c = np.array(ndata_[2])

                ba = a - b
                bc = c - b

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                # print (ndata_, np.degrees(angle))
                angles_.append(np.degrees(angle))
                angle_obj_[np.degrees(angle)] = id_sub

            if len(cdata) > 2:
                a = np.array(cdata[0])
                b = np.array(cdata[1])
                c = np.array(cdata[2])

                ba = a - b
                bc = c - b

                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
                angle = np.arccos(cosine_angle)
                angles.append(np.degrees(angle))
                angle_obj[np.degrees(angle)] = id_sub

    return angles, angle_obj, angles_, angle_obj_

def plot_heatmap(heat_img, kpts):
    # print ('plot heatmap')
    # print (kpts[0])
    for coord in kpts[0]:
        heat_img[coord[1],coord[0]] +=1
    
    # print (np.max(heat_img))
    heat_map = ndimage.filters.gaussian_filter(heat_img, sigma=0.2)
    max_value = np.max(heat_map)
    min_value = np.min(heat_map)
    normalized_heat_map = (heat_map - min_value) / (max_value - min_value)
    
    # print (np.where(normalized_heat_map>=0.5))
    # normalized_heat_map = 255*normalized_heat_map
    # normalized_heat_map = cv2.applyColorMap(normalized_heat_map.astype(np.uint8),cv2.COLORMAP_JET)
    # return heat_img, normalized_heat_map
    return normalized_heat_map

def check_normal(kids,thresh, configs):
    # print (kids)
    # print (thresh)
    # if (0 in kids[0]) and (1 in kids[0]) and (2 in kids[0]) and (4 in kids[0]) and (thresh<4) and not (3 in kids[0]):
    if configs['left_ear'] and (0 in kids[0]) and (1 in kids[0]) and (2 in kids[0]) and (4 in kids[0]) and (thresh<4) and (3 in kids[0]):
        return -1
    elif not (configs['left_ear']) and(0 in kids[0]) and (1 in kids[0]) and (2 in kids[0]) and (4 in kids[0]) and (thresh<4) and not (3 in kids[0]):
        return -1
    else:
        return 1

# def save_video(url, out_name):
#     cap = cv2.VideoCapture(url)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))

#     vid_output = cv2.VideoWriter(
#         out_name, 
#         fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
#         fps=fps, 
#         frameSize=(width, height), 
#         isColor=True
#     )

#     vid_output.write(frame)

def distance(list1, list2):
    """Distance between two vectors."""
    squares = [(p-q) ** 2 for p, q in zip(list1, list2)]
    return sum(squares) ** .5

def check_wrst(ndata,face_kpt,body_kpt):
    wrst_1_dist = distance(ndata[face_kpt],ndata[body_kpt])
    # print (wrst_1_dist)
    if wrst_1_dist<0.15:
        event = 1
    else:
        event = 0
    return event

def check_ear(ndata,kids, face_kpt,body_kpt):
    if face_kpt in kids[0] and body_kpt in kids[0]:
        ear_1_dist = distance(ndata[face_kpt],ndata[body_kpt])
        # print ('ear: ' + str(ear_1_dist))
        return [1, ear_1_dist]
    else:
        return [0,0]
    
    # if wrst_1_dist<0.15:
    #     event = 1
    # else:
    #     event = 0
    # return event
# def run_vid_(vid_path,plot,vis,rotate,plot_kpts, save_video):
def run_vid_(vid_path,configs):

    cap = cv2.VideoCapture(vid_path)

    model = load_model()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    prev = {}
    prev_cent = []
    intv = 10
    dist_plot = []
    frame_ids = []
    plt.show()
    rows = []
    width_rs = None
    height_rs = None
    ctime = []
    events = {}
    cur_event = 0
    prev_event = 0
    starts = 0
    event_id = {}
    ear_event = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (100,500)
    fontScale              = 1
    fontColor              = (0,255,255)
    thickness              = 1
    lineType               = 2

    df = pd.DataFrame(columns=['event_type','starts','ends'])
    out_csv_name = os.path.basename(vid_path).split('.')[0] + '_anomalies.csv'
    df.to_csv(out_csv_name,index=False)

    if configs['save_video']:
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if configs['rotate']:
            width = min(960,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            height = min(960,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        else:
            width = min(960,int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = min(960,int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out_name = os.path.basename(vid_path).split('.')[0] + '_anomalies.mp4'

        vid_output = cv2.VideoWriter(
            out_name, 
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"), 
            fps=fps, 
            frameSize=(960, 704), 
            # frameSize=(height, width),
            isColor=True
        )

    

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ftime = cap.get(cv2.CAP_PROP_POS_MSEC)
            fnumber = cap.get(cv2.CAP_PROP_POS_FRAMES)
            # ctime.append(fnumber)
            prev_event = cur_event
            if configs['rotate']:
                frame = cv2.rotate(frame,cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_size = frame.shape[0]*frame.shape[1]
            output, image = run_vid_inference(model,frame)
            ## find anomaly point. and track
            nimg, kpts, kids = vis_vid_output(output, image, model, configs['plot_kpts'])
            # print (kpts)
            # print (kids)

            # if width_rs is None:
            #     width_rs=nimg.shape[0]; height_rs = nimg.shape[1]; 
            #     heat_img = np.zeros((width_rs, height_rs))
            # heat_img, normalized_heat_map = plot_heatmap(heat_img, kpts)
            
            if len(kpts)>0 and len(kpts[0])>0:
                # heat_img = plot_heatmap(heat_img, kpts)
                x,y,ndata = norm_xy(nimg,kpts,kids)
                right_ear = check_ear(ndata,kids, 2,4)
                left_ear = check_ear(ndata,kids, 1,3)
                if configs['left_ear']:
                    if right_ear[0] ==1 and left_ear[0] == 1:
                        if abs(left_ear[1]-right_ear[1]) > 0.05:
                            # print (right_ear, left_ear,abs(left_ear[1]-right_ear[1]))
                            ear_event = 1
                        else:
                            ear_event = 0
                    else:
                        ear_event = 1
                else:
                    # print (left_ear, right_ear)
                    if left_ear[0] == 1 and right_ear[0] == 0:
                        ear_event = 1
                    elif right_ear[0] ==1 and left_ear[0] == 0:
                        # print (right_ear)
                        if right_ear[1] > 0.15:
                            ear_event = 1
                        else:
                            ear_event = 0
                    elif left_ear[0] == 1 and right_ear[0] == 1:
                        ear_event = 1
                    else:
                        ear_event = 0


                if 4 in kids[0] and 10 in kids[0]:
                    wrist_event = check_wrst(ndata,4,10)
                    if wrist_event:
                        print ('right hands up')
                    elif 3 in kids[0] and 9 in kids[0]:
                        wrist_event = check_wrst(ndata,3,9)
                        if wrist_event:
                            print ('left hands up')

                else:
                    wrist_event = 0

                anglec, angle_obj, anglec_, angle_obj_ = calc_dev(x,y,ndata, kpts, kids)

                id_subs = [[4,2,0],[2,0,1],[0,1,3]]                

                if len(angle_obj_) > 0:
                    rows.append(list(angle_obj_.keys()))
                if len(anglec_)>0:
                    dist_plot.append(anglec_[0])
            else: 
                wrist_event = 0

            dist_plot = dist_plot[-200:]
            # ctime = ctime[-200:]
            pd_dist = pd.Series(dist_plot)
            
            dist_std_ = pd_dist.rolling(window=10).std()
            # ids_number = len(events.keys())
            out_dist = [1 if i>4 else 0 for i in dist_std_]
            # out_dist_ = [(j,1)  for i,j in zip(dist_std_, ctime) if i>4 ]

            ## normal is: 0, 1,2,4 and no spike
            if len(kids)>0:
                event_type = check_normal(kids,dist_std_.values[-1],configs)
                ## print this to show types of events
                # print (event_type, wrist_event, ear_event)
                if event_type == 1 or wrist_event or ear_event:
                    event_wrt = 'event'
                else:
                    event_wrt = 'normal'
                threshval = dist_std_.values[-1]
                
                
                if event_wrt == 'event' and starts == 0:
                    starts = 1; cfrm = fnumber
                    if not cfrm in frame_ids:
                        frame_ids.append(cfrm)
                if event_wrt == 'normal' and starts == 1:
                    starts = 0; cfrm = fnumber
                    if not cfrm in frame_ids:
                        frame_ids.append(cfrm)
                

                ## break up events 
                if len(dist_std_) >10:
                    threshprev = dist_std_.values[-2]
                    if threshval > 4 and threshprev<4 and event_wrt == 'event' and starts == 1:
                        starts = 1; cfrm = fnumber
                        if not cfrm in frame_ids:
                            frame_ids.append(cfrm)
                    # if threshval<=4 and starts == 1:
                    #     starts = 0

                # print (frame_ids)
                if len(frame_ids)>0:
                    event_wrt_1 = event_wrt + " : " + str(int(frame_ids[-1])) + "-" + str(int(fnumber))
                    # print (event_wrt_1)
                    df.loc[len(df)] = [event_wrt,int(frame_ids[-1]), int(fnumber)]
                    if configs['vis']:
                        cv2.putText(nimg,event_wrt_1, bottomLeftCornerOfText, font, 
                        fontScale, fontColor, thickness,lineType)


            if df.shape[0]>500:
                df.to_csv(out_csv_name,index=False,mode='a+',header=False)
                df = pd.DataFrame(columns=['event_type','starts','ends'])

            if configs['vis']:
                cv2.imshow('',nimg)
            if configs['save_video']:
                vid_output.write(nimg)
                
            if configs['plot']:
                plt.plot(dist_std_,'k')
                plt.plot(out_dist,'r')
                plt.draw()
                plt.pause(0.001)
                plt.clf()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    df.to_csv(out_csv_name,index=False,mode='a+',header=False)
    cap.release()
    cv2.destroyAllWindows()


# vid_path = 'videos/aicity.MP4'
vid_path = 'videos/vtti.mp4'
# plot = 0 # set to 1 to plot event time series. red line shows start and end of events
# vis = 1 
# rotate = 0 # set to 1 if image is rotated. 
# plot_kpts = 1 ## set to 1 to show keypoints
# save_video = 0 ## saves video with keypoints and events. 
# left_ear = 1  ## is left ear visible
# right_ear = 1 # is right ear visible
configs = {
    'plot': 0,
    'vis': 1,
    'rotate':1, ## rotate image
    'plot_kpts':1, ## plot key points on image
    'save_video': 0, ## saves video with keypoints and events. 
    'left_ear':0,  ## is left ear visible
}
# print (configs)
# print (configs['save_video'])
# print (configs.plot)
# # run_vid_(vid_path, plot, vis,rotate, plot_kpts, save_video)
run_vid_(vid_path, configs)

## todo
## restart event if thresh is high
