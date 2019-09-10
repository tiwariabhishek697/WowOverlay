import cv2
import face_recognition
from PIL import Image, ImageDraw
import numpy as np
import imutils
from datetime import datetime
from flask import Flask, render_template,request,url_for, jsonify,send_file,send_from_directory
from flask_cors import CORS
app = Flask(__name__,template_folder='templates')
#app=Flask(__name__)
CORS(app)

@app.route("/",methods=['POST','GET'])
def main():
	return render_template('wow_overlay_index.html')


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
	"""
	@brief      Overlays a transparant PNG onto another image using CV2

	@param      background_img    The background image
	@param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
	@param      x                 x location to place the top-left corner of our overlay
	@param      y                 y location to place the top-left corner of our overlay
	@param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

	@return     Background image with overlay on top
	"""

	bg_img = background_img.copy()
	background_width = bg_img.shape[1]
	background_height = bg_img.shape[0]

	if overlay_size is not None:
		img_to_overlay_t = imutils.resize(img_to_overlay_t.copy(), width=overlay_size[1],height=overlay_size[0])

	# Extract the alpha mask of the RGBA image, convert to RGB 
	b,g,r,a = cv2.split(img_to_overlay_t)
	overlay_color = cv2.merge((b,g,r))

	# Apply some simple filtering to remove edge noise
	mask = cv2.medianBlur(a,5)

	h, w, _ = overlay_color.shape
	if x + w/2 > background_width:
	    w = int(w/2+background_width - x)
	    #img_to_overlay_t = img_to_overlay_t[:, :w]

	if x-w/2 < 0:
		w=int((w/2+x))-10
		#img_to_overlay_t = img_to_overlay_t[:, :w]

	if y + h/2 > background_height:
		h=int(background_height-y+h/2)
		#img_to_overlay_t=img_to_overlay_t[:h]

	if y - h/2 < 0:
	    h = int(h/2 + y)
	    #img_to_overlay_t = img_to_overlay_t[:h]

	roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] 

	# Black-out the area behind the logo in our original ROI
	img1_bg = cv2.bitwise_and(roi.copy(),roi.copy(),mask = cv2.bitwise_not(mask))

	# Mask out the logo from the logo image.
	img2_fg = cv2.bitwise_and(overlay_color,overlay_color,mask = mask)

	# Update the original image with our new ROI
	bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

	return bg_img


@app.route('/wow_overlay',methods=['GET','POST'])
def wow_overlay():
	if not (request.files['face'].mimetype[:5]=='image'):
		return jsonify(error="Inputs must be face image!")

	face_img = request.files['face']
	option=request.form.get('option',type=str)
	ts=datetime.timestamp(datetime.now())
	t,s=str(ts).split('.')
	face_img.save("wow_overlay_input_images\\'input_'+{}+'_'+{}+'.jpg'".format(t,s))
	img=cv2.imread("wow_overlay_input_images\\'input_'+{}+'_'+{}+'.jpg'".format(t,s))
	wowhead = cv2.imread('wowhead.png',-1) # -1 loads with transparency
	expnose=cv2.imread('expnose.png',-1)
	left_ear=cv2.imread('left_ear.png',-1)
	right_ear=cv2.imread('right_ear.png',-1)
	joker_nose1=cv2.imread('joker_nose1.png',-1)
	joker_nose2=cv2.imread('joker_nose2.png',-1)
	joker_lips1=cv2.imread('jokerlips1.png',-1)
	joker_lips2=cv2.imread('jokerlips2.png',-1)
	left_flag=cv2.imread('left_flag3.png',-1)
	right_flag=cv2.imread('right_flag3.png',-1)

	#butterfly=cv2.imread('butterfly1.png',-1)
	h1,w1=img.shape[:2]
	h2,w2=wowhead.shape[:2]
	face_landmarks_list=face_recognition.face_landmarks(img)

	#Crown imposing
	if option=='wow crown':
		
		point18=face_landmarks_list[0]['left_eyebrow'][0]
		x_point18,y_point18=point18[0],point18[1]
		point22=face_landmarks_list[0]['left_eyebrow'][4]
		x_point22,y_point22=point22[0],point22[1]
		point23=face_landmarks_list[0]['right_eyebrow'][0]
		x_point23,y_point23=point23[0],point23[1]
		point31=face_landmarks_list[0]['nose_bridge'][3]
		x_point31,y_point31=point31[0],point31[1]
		d1=abs(y_point22-y_point31)
		x_head=int((x_point22+x_point23)/2)+1
		y_head=int(y_point22-d1/2)
		#x,y=x_point31,int(y_point22-d1/2)
		d2=abs(face_landmarks_list[0]['chin'][1][0]-face_landmarks_list[0]['chin'][15][0])
		print(d2)
		overlay_w1=int(1.5*d2)
		overlay_h1=d1
		

		crowned=overlay_transparent(img, wowhead, x_head, y_head, (overlay_h1,overlay_w1))
		ts1=datetime.timestamp(datetime.now())
		t1,s1=str(ts1).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t1,s1),crowned)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t1,s1),mimetype='image/png')


	#expnose imposing
	if option=='exp nose':
		point34=face_landmarks_list[0]['nose_tip'][2]
		x_point34,y_point34=point34[0],point34[1]
		point31=face_landmarks_list[0]['nose_bridge'][3]
		x_point31,y_point31=point31[0],point31[1]
		x_nose,y_nose=int((x_point31+x_point34)/2),int((y_point31+y_point34)/2)
		d2=abs(face_landmarks_list[0]['chin'][1][0]-face_landmarks_list[0]['chin'][15][0])
		overlay_w2=int(0.5*d2)
		overlay_h2=abs(face_landmarks_list[0]['chin'][1][1]-face_landmarks_list[0]['chin'][2][1])
		#img_dst = cv2.warpAffine(expnose, rotation_matrix, size,
        #                     flags=cv2.INTER_LINEAR,
        #                     borderMode=cv2.BORDER_TRANSPARENT)

		temp_result=overlay_transparent(img, expnose, x_nose, y_nose, (overlay_h2,overlay_w2))
		ts2=datetime.timestamp(datetime.now())
		t2,s2=str(ts2).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')

	#laef ear imposing
	if option=='leaf ear':
		point1=face_landmarks_list[0]['chin'][0]
		point17=face_landmarks_list[0]['chin'][16]
		d3=abs(face_landmarks_list[0]['nose_bridge'][0][1]-face_landmarks_list[0]['nose_bridge'][1][1])
		x_left_ear=point1[0]-d3
		y_left_ear=point1[1]
		x_right_ear=point17[0]+d3
		y_right_ear=point17[1]
		overlay_w3=2*abs(face_landmarks_list[0]['nose_bridge'][0][1]-face_landmarks_list[0]['nose_bridge'][2][1])
		overlay_h3=2*abs(face_landmarks_list[0]['nose_bridge'][0][1]-face_landmarks_list[0]['nose_bridge'][3][1])
		
		temp_result1=overlay_transparent(img, left_ear, x_left_ear, y_left_ear, (overlay_h3,overlay_w3)) #Replace left_ear with img_dst1 
		temp_result2=overlay_transparent(temp_result1, right_ear, x_right_ear, y_right_ear, (overlay_h3,overlay_w3)) #Replace right_ear with img_dst2
		ts3=datetime.timestamp(datetime.now())
		t3,s3=str(ts3).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t3,s3),temp_result2)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t3,s3),mimetype='image/png')

	if option=='joker nose1':
		point34=face_landmarks_list[0]['nose_tip'][2]
		x_point34,y_point34=point34[0],point34[1]
		point31=face_landmarks_list[0]['nose_bridge'][3]
		x_point31,y_point31=point31[0],point31[1]
		x_nose,y_nose=int((x_point31+x_point34)/2),int((y_point31+y_point34)/2)
		d2=abs(face_landmarks_list[0]['chin'][1][0]-face_landmarks_list[0]['chin'][15][0])
		overlay_w2=int(0.3*d2)
		overlay_h2=abs(face_landmarks_list[0]['chin'][1][1]-face_landmarks_list[0]['chin'][2][1])
		

		temp_result=overlay_transparent(img, joker_nose1, x_point31, y_point31, (overlay_h2,overlay_w2))
		ts2=datetime.timestamp(datetime.now())
		t2,s2=str(ts2).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')

	if option=='joker nose2':
		point34=face_landmarks_list[0]['nose_tip'][2]
		x_point34,y_point34=point34[0],point34[1]
		point31=face_landmarks_list[0]['nose_bridge'][3]
		x_point31,y_point31=point31[0],point31[1]
		x_nose,y_nose=int((x_point31+x_point34)/2),int((y_point31+y_point34)/2)
		d2=abs(face_landmarks_list[0]['chin'][1][0]-face_landmarks_list[0]['chin'][15][0])
		overlay_w2=int(0.3*d2)
		overlay_h2=abs(face_landmarks_list[0]['chin'][1][1]-face_landmarks_list[0]['chin'][2][1])
		
		temp_result=overlay_transparent(img, joker_nose2, x_point31, y_point31, (overlay_h2,overlay_w2))
		ts2=datetime.timestamp(datetime.now())
		t2,s2=str(ts2).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')

	if option=='joker lips1':
		point63=face_landmarks_list[0]['top_lip'][9]
		x_point63,y_point63=point63[0],point63[1]
		point67=face_landmarks_list[0]['bottom_lip'][3]
		x_point67,y_point67=point67[0],point67[1]
		x_lip,y_lip=face_landmarks_list[0]['nose_tip'][2][0],face_landmarks_list[0]['nose_tip'][2][1]
		overlay_w=int(3*abs(face_landmarks_list[0]['top_lip'][0][0]-face_landmarks_list[0]['top_lip'][6][0]))
		overlay_h=abs(face_landmarks_list[0]['top_lip'][3][1]-face_landmarks_list[0]['bottom_lip'][9][1])
		
		temp_result=overlay_transparent(img,joker_lips1 , x_lip, y_lip, (overlay_h,overlay_w))
		ts2=datetime.timestamp(datetime.now())
		t2,s2=str(ts2).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')


	if option=='joker lips2':
		point63=face_landmarks_list[0]['top_lip'][9]
		x_point63,y_point63=point63[0],point63[1]
		point67=face_landmarks_list[0]['bottom_lip'][3]
		x_point67,y_point67=point67[0],point67[1]
		x_lip,y_lip=int((x_point67+x_point63)/2),int((y_point67+y_point63)/2)-2
		overlay_w=int(2*abs(face_landmarks_list[0]['top_lip'][0][0]-face_landmarks_list[0]['top_lip'][6][0]))
		overlay_h=abs(face_landmarks_list[0]['top_lip'][3][1]-face_landmarks_list[0]['bottom_lip'][9][1])
		
		temp_result=overlay_transparent(img,joker_lips2, x_point63, y_point63, (overlay_h,overlay_w))
		ts2=datetime.timestamp(datetime.now())
		t2,s2=str(ts2).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')

	if option=='indian national flag':
		point_42=face_landmarks_list[0]['left_eye'][5]
		x_42,y_42=point_42[0],point_42[1]
		point_4=face_landmarks_list[0]['chin'][3]
		x_4,y_4=point_4[0],point_4[1]
		point_32=face_landmarks_list[0]['nose_tip'][0]
		x_32,y_32=point_32[0],point_32[1]
		left_x_b=int((x_42 + 2*x_4 + x_32)/4)
		left_y_b=int((y_42 + 2*y_4 + y_32)/4)
		
		point_28=face_landmarks_list[0]['nose_bridge'][0]
		point_31=face_landmarks_list[0]['nose_bridge'][2]
		overlay_w=abs(x_32 - face_landmarks_list[0]['chin'][2][0])
		overlay_h=abs(point_28[1]-point_31[1])
		
		temp_result=overlay_transparent(img, left_flag, left_x_b, left_y_b, (overlay_h,overlay_w))
		#ts2=datetime.timestamp(datetime.now())
		#t2,s2=str(ts2).split('.')
		#cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result)
		#return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')


		point_47=face_landmarks_list[0]['right_eye'][4]
		x_47,y_47=point_47[0],point_47[1]
		point_14=face_landmarks_list[0]['chin'][13]
		x_14,y_14=point_14[0],point_14[1]
		point_36=face_landmarks_list[0]['nose_tip'][4]
		x_36,y_36=point_36[0],point_36[1]
		right_x_b=int((x_47 + 2*x_14 + x_36)/4)
		right_y_b=int((y_47 + 2*y_14 + y_36)/4)
		
		temp_result1=overlay_transparent(temp_result, right_flag, right_x_b, right_y_b, (overlay_h,overlay_w))
		ts2=datetime.timestamp(datetime.now())
		t2,s2=str(ts2).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),temp_result1)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t2,s2),mimetype='image/png')





	#All wowexp filter imposing
	if option=='wowexp':
		#crown part
		
		point18=face_landmarks_list[0]['left_eyebrow'][0]
		x_point18,y_point18=point18[0],point18[1]
		point22=face_landmarks_list[0]['left_eyebrow'][4]
		x_point22,y_point22=point22[0],point22[1]
		point23=face_landmarks_list[0]['right_eyebrow'][0]
		x_point23,y_point23=point23[0],point23[1]
		point31=face_landmarks_list[0]['nose_bridge'][3]
		x_point31,y_point31=point31[0],point31[1]
		d1=abs(y_point22-y_point31)
		x_head=int((x_point22+x_point23)/2)+1
		y_head=int(y_point22-d1/2)
		#x,y=x_point31,int(y_point22-d1/2)
		d2=abs(face_landmarks_list[0]['chin'][1][0]-face_landmarks_list[0]['chin'][15][0])
		print(d2)
		overlay_w1=int(1.5*d2)
		overlay_h1=d1
		

		crowned=overlay_transparent(img, wowhead, x_head, y_head, (overlay_h1,overlay_w1))
		
		#nose part
		point34=face_landmarks_list[0]['nose_tip'][2]
		x_point34,y_point34=point34[0],point34[1]
		x_nose,y_nose=int((x_point31+x_point34)/2),int((y_point31+y_point34)/2)
		overlay_w2=int(0.5*d2)
		overlay_h2=abs(face_landmarks_list[0]['chin'][1][1]-face_landmarks_list[0]['chin'][2][1])
		

		temp_result=overlay_transparent(crowned, expnose, x_nose, y_nose, (overlay_h2,overlay_w2))

		#ear part
		point1=face_landmarks_list[0]['chin'][0]
		point17=face_landmarks_list[0]['chin'][16]
		d3=abs(face_landmarks_list[0]['nose_bridge'][0][1]-face_landmarks_list[0]['nose_bridge'][1][1])
		x_left_ear=point1[0]-d3
		y_left_ear=point1[1]
		x_right_ear=point17[0]+d3
		y_right_ear=point17[1]
		overlay_w3=2*abs(face_landmarks_list[0]['nose_bridge'][0][1]-face_landmarks_list[0]['nose_bridge'][2][1])
		overlay_h3=2*abs(face_landmarks_list[0]['nose_bridge'][0][1]-face_landmarks_list[0]['nose_bridge'][3][1])
		

		temp_result1=overlay_transparent(temp_result, left_ear, x_left_ear, y_left_ear, (overlay_h3,overlay_w3))
		result=overlay_transparent(temp_result1, right_ear, x_right_ear, y_right_ear, (overlay_h3,overlay_w3))
		ts3=datetime.timestamp(datetime.now())
		t3,s3=str(ts3).split('.')
		cv2.imwrite("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t3,s3),result)
		return send_file("wow_overlay_output_images\\'output_'+{}+'_'+{}+'.png'".format(t3,s3),mimetype='image/png')

	
if __name__=='__main__':
	app.run(host='0.0.0.0',debug=True,port=8088)
