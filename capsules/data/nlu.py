import numpy as np
import os
import pickle

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit([{'sphere', 'Eggplant', 'Broccoli', 'Parsley', 'jalapeno', 'Garlic', 'bell_pepper', 'Onion', 'cube', 'chilli', 'cucumber', 'Potato', 'Onion_green', 'Radish', 'tomato', 'lettuce', 'cabbage', 'Carrot', 'cheese', 'cylinder'}])

def bbox_rearrange(tree,boxes= [],classes={},all_classes=[]):
	for i in range(0, tree.num_children):
		updated_tree,boxes,classes,all_classes = bbox_rearrange(tree.children[i],boxes=boxes,classes=classes,all_classes=all_classes)
		tree.children[i] = updated_tree     
	if tree.function == "describe":
		xmax,ymax,zmin,xmin,ymin,zmax = tree.bbox_origin
		box = np.array([xmin,ymin,zmin,xmax,ymax,zmax])
		tree.bbox_origin = box
		boxes.append(box)
		classes["shape"] = tree.word
		all_classes.append(classes)
		classes = {}
	if tree.function == "combine":
		if "large" in tree.word or "small" in tree.word:
			classes["size"] = tree.word
		elif "metal" in tree.word or "rubber" in tree.word:
			classes["material"] = tree.word
		else:
			classes["color"] = tree.word
	return tree,boxes,classes,all_classes

def return_labels(tree_file_seq, only_one=False):
	tree = pickle.load(open(os.path.join("/home/mprabhud/dataset/clevr_veggies",tree_file_seq),"rb"))
	tree,boxes,_,all_classes = bbox_rearrange(tree,boxes=[],classes={},all_classes=[])
	shapes_present = {class_val["shape"] for class_val  in all_classes}
	if onle_one:
		labels_present = [shapes_present]	
	else:
		labels_present = []
		for i in range(40):
			labels_present.append(shapes_present)
	return mlb.transform(labels_present)

# do_shape = 
# do_color = 
# do_material = 
# do_style = 
# do_style_content = 
# do_color_content = 
# do_material_content = 


# def trees_rearrange(trees):
# 	updated_trees =[]
# 	all_bboxes = []
# 	all_scores = []
# 	all_classes_list = []
# 	for tree in trees:
# 		tree,boxes,_,all_classes = bbox_rearrange(tree,boxes=[],classes={},all_classes=[])
# 		if do_shape:
# 			classes = [class_val["shape"] for class_val  in all_classes]
# 		elif do_color:
# 			classes = [class_val["color"] for class_val  in all_classes]
# 		elif do_material:
# 			classes = [class_val["material"] for class_val  in all_classes]
# 		elif do_style:
# 			classes = [class_val["color"]+"_"+ class_val["material"] for class_val  in all_classes]
# 		elif do_style_content:
# 			classes = [class_val["shape"]+"/"+class_val["color"]+"_"+ class_val["material"] for class_val  in all_classes]
# 		elif do_color_content:            
# 			classes = [class_val["shape"]+"/"+class_val["color"] for class_val  in all_classes]
# 		elif do_material_content:            
# 			classes = [class_val["shape"]+"/"+ class_val["material"] for class_val  in all_classes]
# 		else:            
# 			classes = [class_val["shape"]+"/"+ class_val["color"] +"_"+class_val["material"] for class_val  in all_classes]
# 		boxes = np.stack(boxes)
# 		classes = np.stack(classes)
# 		N,_  = boxes.shape 
# 		assert N == len(classes)
# 		scores = np.pad(np.ones([N]),[0,hyp.N-N])
# 		boxes = np.pad(boxes,[[0,hyp.N-N],[0,0]])
# 		classes = np.pad(classes,[0,hyp.N-N])
# 		updated_trees.append(tree)
# 		all_classes_list.append(classes)
# 		all_scores.append(scores)
# 		all_bboxes.append(boxes)
# 	all_bboxes = np.stack(all_bboxes)
# 	all_scores = np.stack(all_scores)
# 	all_classes_list = np.stack(all_classes_list)
# 	return all_bboxes,all_scores,all_classes_list


# def get_max_classes(trees):
# 	shape_set = set()
# 	color_set = set()
# 	size_set = set()
# 	material_set = set()
# 	max_obj = 0

# 	for tree in trees:
# 		tree,boxes,_,all_classes = bbox_rearrange(tree,boxes=[],classes={},all_classes=[])
# 		max_obj = max(max_obj, len(all_classes))
# 		shape_set.update([class_val["shape"] for class_val  in all_classes])
# 		color_set.update([class_val["color"] for class_val  in all_classes])
# 		material_set.update([class_val["material"] for class_val  in all_classes])
# 		size_set.update([class_val["size"] for class_val  in all_classes])

# 	return shape_set, color_set, size_set, material_set, max_obj


