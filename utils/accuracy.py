def topk_acc(score_mat, label, k=1):
	max_idx = score_mat.topk(k, dim=1)
	correct_cnt = 0
	total_cnt = len(max_idx[1])
	for i in range(total_cnt):
		# max_idx[1][i][1]
		# 第一个 1 指的是 topk的序号（0对应的是值）
		# 第二个 i 指的是 第i个样本
		# max_idx[1][i]是一个list，表示topk的元素对应的标签值序列
		if label[i] in label[max_idx[1][i]]:
			correct_cnt += 1
	return correct_cnt/total_cnt
