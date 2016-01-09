import pandas as pd
import numpy as np
import math
import time
import pickle
import copy
import random
from haversine import haversine

class Tabu:
	def __init__(self, code, gift1, gift2) :
		self.code = code
		self.gift1 = gift1
		self.gift2 = gift2
		self.level = 50
	def kill(self) :
		self.level -= 1
		if self.level==0 :
			return True
		return False
	def compare(self, code, gift1, gift2) :
		if self.code==code and self.gift1 == gift1 and self.gift2 == gift2 : return True
		else : return False

def dist_north(id) :
	# calculates the haversine distance between north pole and gift[id]
	return haversine((gift_array[id,0]*180.0/math.pi, gift_array[id,1]*180.0/math.pi), (90.0, 0.0))

def dist(id1, id2) :
	# calculates the haversine distance between gift[id1] and gift[id2]
	return haversine((gift_array[id1,0]*180.0/math.pi, gift_array[id1,1]*180.0/math.pi), (gift_array[id2,0]*180.0/math.pi, gift_array[id2,1]*180.0/math.pi))

# (1) weight big (2) located in upper region (high latitude)
def compare(idx1, idx2) :
	if gift_weight_list[idx1] != gift_weight_list[idx2] :
		if (gift_weight_list[idx1] < gift_weight_list[idx2]) : return 1
		else : return -1
	if gift_array[idx1,0] < gift_array[idx2,0] : return 1
	elif gift_array[idx1,0] == gift_array[idx2,0] : return 0
	else : return -1
	
def bin_search(list, idx) :
	if compare(list[-1], idx)<0 : return len(list)
	left = 0
	right = len(list)-1
	mid = (left+right)/2
	while left<=right :
		mid = (left+right)/2
		if compare(list[mid], idx)<0 :
			left = mid+1
		elif compare(list[mid], idx)>0 : 
			right = mid-1
		else :
			return mid
	if right<mid : return mid
	else : return mid+1

def sofar(gift1, clu2) :
	if len(cluster_list[clu2])==0 : return False
	gift2 = cluster_list[clu2][0]
	difflat = abs(gift_array[gift1,0]-gift_array[gift2,0])*180.0/math.pi
	temp = abs(gift_array[gift1,1]-gift_array[gift2,1])*180.0/math.pi
	difflong = min(temp, 360.0-temp)
	if difflat + difflong > 60.0 : return True
	else : return False

def dist_calc(mlist, dist_dic, initial_weight) :
	nowposition = -1
	nowweight = initial_weight
	nowsum = 0.0
	for x in mlist :
		nowdist = dist_dic[(nowposition, x)]
		nowsum += nowweight*nowdist
		nowweight -= gift_weight_list[x]
		nowposition = x
	nowsum += nowweight * dist_dic[(-1, x)]
	return nowsum

gifts = pd.read_csv("gifts.csv") # 0~99999

# save values from dataframe
gift_array = np.zeros((gifts.shape[0], 2))
gift_weight_list = np.zeros((gifts.shape[0],))
for i in range(gifts.shape[0]) :
	# convert lat/lon into radian
	gift_array[i, 0] = gifts.loc[i, 'Latitude']*math.pi/180.0
	gift_array[i, 1] = gifts.loc[i, 'Longitude']*math.pi/180.0
	gift_weight_list[i] = gifts.loc[i, 'Weight']

results = pd.read_csv("result_new_martini_46.csv") # put continue_file_name.csv

k=1523	
cluster_list = []
accum_dist_list = [] # total travel distance to get to 'certain' gift place
accum_weight_list = [] # weight after giving gift to 'certain' gift place
WRW_list = np.zeros((k, ))
weight_list = np.zeros((k, ))
for i in range(k) : cluster_list.append([]); accum_dist_list.append([]); accum_weight_list.append([])
for i in range(gifts.shape[0]) : cluster_list[results.loc[i, 'TripId']-1].append(results.loc[i, 'GiftId']-1)

total_sum = 0.0
for i in range(k) :
	#cluster_list[i] = sorted(cluster_list[i], cmp=compare)
	nowsum = 0.0
	nowweight = 10.0
	for x in cluster_list[i] : nowweight+=gift_weight_list[x]
	weight_list[i] = nowweight
	nowposition = (90.0, 0.0)
	for x in cluster_list[i] :
		nowdist = haversine(nowposition, (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi))
		nowsum += nowweight*nowdist
		nowweight -= gift_weight_list[x]
		nowposition = (gift_array[x,0]*180.0/math.pi, gift_array[x,1]*180.0/math.pi)
		accum_dist_list[i].append(nowdist)
		accum_weight_list[i].append(nowweight)
	for j in range(len(cluster_list[i])) : 
		if j==0 : continue
		accum_dist_list[i][j] = accum_dist_list[i][j] + accum_dist_list[i][j-1]

	nowsum += nowweight*haversine(nowposition, (90.0, 0.0))
	WRW_list[i] = nowsum;
	total_sum += nowsum;
print total_sum

total_changed = 0.0

for cluster_num in range(k) :
	print "Fixing trip number %d" % cluster_num
	print "Initial WRW : %lf" % WRW_list[cluster_num]
	nowlist = cluster_list[cluster_num]

	# calculate haversine between gifts
	dist_dict = {}
	gift_num = len(nowlist)
	for i in range(gift_num) :
		for j in range(gift_num) : 
			dist_dict[(nowlist[i], nowlist[j])] = dist(nowlist[i], nowlist[j])
	for i in range(gift_num) : 
		dist_dict[(-1, nowlist[i])] = dist_north(nowlist[i])

	total_result = WRW_list[cluster_num]
	total_best_result = WRW_list[cluster_num]
	total_best_list = []
	tabu_list = []

	nondec = 0.0
	for epoch in range(300) : 

		best_result = 98765432123456789.0
		best_i = -1
		best_j = -1
		best_code = 0
		gift_num = len(nowlist)

		code = 0
		# neighbor : change two randomly
		for i in range(gift_num) :
			for j in range(gift_num) :
				if i<=j : continue
				tocontinue = False
				for tabu in tabu_list :
					if tabu.compare(code, i, j) == True : tocontinue=True; break;
				if tocontinue : continue
				nowlist[i], nowlist[j] = nowlist[j], nowlist[i]
				now_result = dist_calc(nowlist, dist_dict, weight_list[cluster_num])
				nowlist[i], nowlist[j] = nowlist[j], nowlist[i]
				if best_result > now_result : 
					best_result = now_result
					best_i = i
					best_j = j
					best_code = code

		code = 1
		# neighbor : move one to other place : pop i, push to j
		for i in range(gift_num) :
			for j in range(gift_num) :
				if i==j : continue
				tocontinue = False
				for tabu in tabu_list :
					if tabu.compare(code, i, j) == True : tocontinue=True; break;
				if tocontinue : continue
				giftid = nowlist[i]
				nowlist.pop(i)
				nowlist.insert(j, giftid)
				now_result = dist_calc(nowlist, dist_dict, weight_list[cluster_num])
				nowlist.pop(j)
				nowlist.insert(i, giftid)

				if best_result > now_result : 
					best_result = now_result
					best_i = i
					best_j = j
					best_code = code

		# real move
		if total_best_result > best_result : print "Epoch %d : %lf, delta : %lf, best_loss : %lf" % (epoch, best_result, best_result - total_result, total_best_result - WRW_list[cluster_num])
		total_result = best_result
		if total_best_result > total_result :
			total_best_result = total_result
			total_best_list = copy.deepcopy(nowlist)
			nondec = 0
		else :
			nondec+=1
			if nondec>30 : break
		if best_code==0 : nowlist[best_i], nowlist[best_j] = nowlist[best_j], nowlist[best_i]
		else :
			giftid = nowlist[best_i]
			nowlist.pop(best_i)
			nowlist.insert(best_j, giftid)

		tabu_list.append(Tabu(best_code, best_i, best_j))
		tabu_list.append(Tabu(best_code, best_j, best_i))

		tabu_list = [x for x in tabu_list if not x.kill()]

	cluster_list[cluster_num] = total_best_list
	total_changed += (total_best_result-WRW_list[cluster_num])
	print "Total Changed : %lf" % total_changed

	if cluster_num%400!=0 or cluster_num==0 : continue
	final_data = pd.DataFrame(index=np.arange(gifts.shape[0]), columns=['GiftId', 'TripId'])
	cnt = 0
	for i in range(k) :
		for x in cluster_list[i] :
			final_data.loc[cnt, 'GiftId'] = x+1
			final_data.loc[cnt, 'TripId'] = i+1
			cnt+=1
	final_data.to_csv("result_stage2.csv", encoding='utf-8', index=False)
	print "Done until : %d, Total Score : %lf" % (cluster_num, total_sum + total_changed)

