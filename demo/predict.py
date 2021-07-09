#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : zhaoguanhua
@Email   : 
@Time    : 2021/7/8 16:56
@File    : predict.py
@Software: PyCharm
"""


import os,glob
import numpy as np
import torch
import segmentation_model as smp
import time
import gdal

def data_preprocess(input_array):
    """
    数据预处理
    :param input_array: 分块数据列表
    :return:
    """
    #归一化
    input_array/=255.0

    mean=np.array([0.485,0.456,0.406])
    std=np.array([0.229,0.224,0.225])

    input_array-=mean
    input_array/=std

    #通道顺序转换
    input_array=np.transpose(input_array,(0,3,1,2))

    #numpy转tensor
    input_array=torch.from_numpy(input_array)
    input_array=input_array.to(DEVICE)

    return input_array

def predict_by_gdal(origin_file,predict_file,model,batch_size):
    """
    使用gdal读写影像，分块输入模型进行推理
    :param origin_file: 输入图像路径
    :param predict_file: 输出图像路径
    :param model: 模型
    :param batch_size:batch_size大小
    :return:
    """
    gdal.AllRegister()
    dataset=gdal.Open(origin_file)
    #获取影像基本信息
    image_width=dataset.RasterXSize   #宽
    image_height=dataset.RasterYSize  #高
    geos=dataset.GetGeoTransform()    #仿射变换矩阵
    proj=dataset.GetProjection()      #投影信息

    #创建输出数据集
    driver=gdal.GetDriverByName("Gtiff")
    outdataset=driver.Create(predict_file,image_width,image_height,1,gdal.GDT_Byte)
    # outdataset.SetGeoTransform()
    # outdataset.SetProjection()
    output_band=outdataset.GetRasterBand(1)

    clip_size=512              #模型接收图片的尺寸
    stride_size=clip_size//2   #窗口滑动距离
    drop_size=(clip_size-stride_size)//2   #最终结果保存时每边丢弃的像素尺寸


    tl_height=0   #小图片左上角在原始图像上的行号
    br_height=0   #小图片右下角在原始图片上的行号

    #分块读取
    positions=[]
    clip_datas=[]

    while (tl_height+stride_size<image_height) or (br_height<image_height):
        br_height=tl_height+clip_size
        if(br_height>image_height): #判断切割的高度是否越界
            br_height=image_height
        tl_width=0  #小图片左上角在原始图片上的列号
        br_width=0  #小图片右下角在原始图片上的列号
        while (tl_width+stride_size<image_width) or (br_width<image_width):
            br_width=tl_width+clip_size
            if (br_width>image_width): #判断切割宽度是否越界
                br_width=image_width

            block_height=br_height-tl_height
            block_width=br_width-tl_width

            #使用dataset读取数据
            clip_data=dataset.ReadAsArray(xoff=tl_width,yoff=tl_height,xsize=block_width,ysize=block_height)
            if np.min(clip_data)<255: #判断小图片最小值是否是255(无效值区域)
                pad_right=clip_size-block_width
                pad_bottom=clip_size-block_height
                if pad_bottom!=0 or pad_right!=0: #小图片尺寸不满足clip_size*clip_size,即滑动到了大图像的边界，需要填充
                    # clip_data[:3,:,;]遥感影像可能有多个波段，直接使用dataset读取会获取所有波段数据，推理时只保留前三个
                    clip_data=np.pad(clip_data[:3,:,:],((0,0),(0,pad_bottom),(0,pad_right)),"constant",constant_values=(255,255))

                clip_datas.append(clip_data)

                positions.append([tl_height,tl_width,br_height,br_width]) #记录小图片在大图片上的实际位置
            else:
                #小图片全为无效值
                predict=np.zeros((block_height,block_width))
                output_band.WriteArray(predict,xoff=tl_width,yoff=tl_height)

            if len(clip_datas)==batch_size or ((br_height==image_height) and (br_width==image_width) and len(positions)!=0):

                #数据预处理
                clip_tensor=data_preprocess(clip_datas)

                predict=model.predict(clip_tensor.float())
                predict=(predict.squeeze(dim=1).cpu().numpy())
                predict=np.argmax(predict,axis=1)

                save_clip_result(predict,positions,output_band,image_height,image_width,drop_size)
                positions.clear()
                clip_datas.clear()

            tl_width+=stride_size

        tl_height+=stride_size

    del dataset
    del outdataset


def save_clip_result(result,positions,output_band,image_height,image_width,drop_size):
    """
    将分块推理的结果按位置保存在原图尺寸的输出结果上
    :param result: 推理结果
    :param positions: 位置信息
    :param output_band: 输出波段
    :param image_height: 原图高
    :param image_width: 原图宽
    :param drop_size: 每个小图片的输出结果中每条边丢弃的像素尺寸
    :return:
    """

    block_num=result.shape[0]

    for batch_id in range(block_num):
        drop_up=drop_size
        drop_down=drop_size
        drop_left=drop_size
        drop_right=drop_size
        tl_height,tl_width,br_height,br_width=positions[batch_id]

        if tl_height==0:
            #小图片位于大图片的最上侧，小图片上侧不丢弃
            drop_up=0
        if tl_width==0:
            # 小图片位于大图片的最左侧，小图片左侧不丢弃
            drop_left=0
        if br_height==image_height:
            # 小图片位于大图片的最下侧，小图片下侧不丢弃
            drop_down=0
        if br_width==image_width:
            # 小图片位于大图片的最右侧，小图片右侧不丢弃
            drop_right=0

        save_height=br_height-tl_height-drop_up-drop_down
        save_width=br_width-tl_width-drop_left-drop_right

        output_band.WriteArray(result[batch_id,drop_up:drop_up+save_height,drop_left:drop_left+save_width],
                               xoff=tl_width+drop_left,yoff=tl_height+drop_up)







if __name__ == '__main__':
    model_path=""
    use_cuda=torch.cuda.is_available()
    DEVICE=torch.device("cuda:0" if use_cuda else "CPU")

    n_classes=5
    #加载网络
    model = smp.PointRend(
        smp.deeplabv3(pretrained=True, resnet="res50", num_classes=n_classes),
        smp.PointHead(in_c=512 + n_classes, num_classes=n_classes))

    model.load_state_dict(torch.load(model_path,map_location=DEVICE))
    if isinstance(model,torch.nn.DataParallel):
        model=model.module
    model=model.to(DEVICE)
    model=model.eval()

    origin_dir=r""
    output_dir=r""
    os.makedirs(output_dir,exist_ok=True)

    origin_images=glob.glob(os.path.join(origin_dir,"*.PNG"))

    for origin_file in origin_images:
        origin_image=os.path.basename(origin_file)
        origin_path=os.path.join(origin_dir,origin_image)
        output_path=os.path.join(output_dir,origin_image).replace(".PNG",".tiff")
        print(origin_path)
        start_t=time.time()
        predict_by_gdal()
        end_t=time.time()
        print("{file} use time: {time} s".format(file=origin_image,time=(end_t-start_t)))