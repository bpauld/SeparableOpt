import numpy as np
import scipy as sp


class CaratheodoryOutput:
    def __init__(self, y_dic:dict[int, list[np.ndarray]]):
        for i, pair in y_dic.items():
            if len(pair) != 2:
                raise ValueError(f"For key '{i}', pair does not have length 2")
            if not isinstance(pair[0], np.ndarray):
                raise ValueError(f"For key '{i}', the first elememt is not a numpy array")
            if not isinstance(pair[1], np.ndarray):
                raise ValueError(f"For key '{i}', the second element is not a numpy array")
            if pair[0].shape[1] != pair[1].shape[0]:
                raise ValueError(f"For key '{i}', the second dimension of the array must match the length of the second element of the list.")
        self.y_dic = y_dic    


class CaratheodoryMNPSolve():
    """
    Class to compute the approximate Caratheodory decomposition
    """

    def __init__(self,
                 y_dic,
                 verbose:bool = True):
        self.y_dic = y_dic
        self.y_dic_integer_indexed = self.compute_integer_indexed_y_dic()
        self.conversion_dict = self.compute_conversion_dict() #to help 
        self.m = self.y_dic_integer_indexed[0].shape[0]  # Assuming all y_dic[i] have the same shape
        self.n = len(y_dic)
        self.verbose = verbose


    def compute_integer_indexed_y_dic(self):
        new_y_dic = {}
        for i, (key, pair) in enumerate(self.y_dic.items()):
            new_y_dic[i] = pair
        return new_y_dic

    def compute_conversion_dict(self):
        conversion_dict = {}
        for i, original_indexed in enumerate(self.y_dic):
            conversion_dict[i] = original_indexed
        return conversion_dict
    
    def convert_dict_to_original_indexing(self, dic):
        new_dic = {}
        #convert an integer-indexed dictionary to the original indexing
        for i, (key, pair) in enumerate(dic.items()):
            assert i == key
            new_dic[self.conversion_dict[i]] = pair
        return new_dic


    def initialize_PS(self, i, k, z_star):
        P_S = np.zeros((self.m, 1))
        P_S[:, 0] = self.y_dic_integer_indexed[i][:, k] - z_star[:self.m]
        P_S_index_list = [i]
        return P_S, P_S_index_list
    
    def build_vector_from_indices(self, i, k):
        v = np.zeros(self.m + self.n)
        v[:self.m] = self.y_dic_integer_indexed[i][:, k]
        v[self.m + i] = 1
        return v
    
    def compute_PS_dot_v(self, v, P_S, P_S_index_list, z_star):
        res = np.zeros(self.n + self.m)
        res[0:self.m] = P_S @ v
        for (index_in_list, i) in enumerate(P_S_index_list):
            res[self.m + i] += v[index_in_list]
        res[self.m:] -= z_star[self.m:]
        return res
    
    def compute_PS_T_dot_v(self, v, P_S, P_S_index_list, z_star):
        #res = np.zeros(P_S.shape[1])
        res = P_S.T @ v[:self.m]
        res += v[self.m:][P_S_index_list]
        res = res - z_star[self.m:].dot(v[self.m:])
        return res
    
    def update_PS(self, P_S, P_S_index_list, i, k, z_star):
        P_i_k = self.y_dic_integer_indexed[i][:, k] - z_star[0:self.m]
        P_S = np.c_[P_S, P_i_k]
        P_S_index_list.append(i)
        return P_S, P_S_index_list
    
    def lmo(self, grad):
        best_index = (-1, -1)
        best_value = np.inf
        for i in range(self.n):
            candidates = self.y_dic_integer_indexed[i].T @ grad[:self.m] + grad[self.m + i]
            min_candidate_index = np.argmin(candidates)
            min_candidate_value = candidates[min_candidate_index]
            if min_candidate_value < best_value:
                best_value = min_candidate_value
                best_index = (i, min_candidate_index)
        return best_index[0], best_index[1]


    # Implementation of Wolfe's min point norm algorithm from paper
    # "FINDING THE NEAREST POINT IN A POLYTOPE" 
    # using the sparsity of the vectors in A_K to never form the matrix P[S] 
    # and using the implementation trick D from the paper to only ever solve triangular linear systems.
    def solve(self, z, T, Z1=1e-12, Z2=1e-10, Z3=1e-10):
        active_set = []
        is_index_added = np.zeros(self.n)

        #compute z_star for convenience
        z_star = np.zeros(self.m + self.n)
        z_star[:self.m] = z / self.n
        z_star[self.m:] = np.ones(self.n) / self.n


        best_index = (-1, -1)
        best_value = np.inf

        for i in range(self.n):
            for k in range(self.y_dic_integer_indexed[i].shape[1]):
                P_i_k = self.y_dic_integer_indexed[i][:, k] - z_star[:self.m]
                candidate = P_i_k.dot(P_i_k)

                if candidate < best_value:     
                    best_value = candidate
                    best_index = (i, k)
        best_i, best_k = best_index
        is_index_added[best_i] = 1


        active_set = [(best_i, best_k)]
        opt_lambda = np.ones(1)
        #P_S = np.zeros((n+m+1, 1))
        #P_S[:, 0] = build_vector_from_indices(best_i, best_k, y_dic, prob) - z_star
        P_S_small, P_S_index_list = self.initialize_PS(best_i, best_k, z_star)
        
        #R = np.sqrt(1 + np.linalg.norm(P_S[:, 0])**2) * np.ones((1, 1)) 
        #preallocate the max size of R
        P_S_0 = self.build_vector_from_indices(best_i, best_k) - z_star
        R_big = np.zeros((self.n+self.m, self.n+self.m)) #prob never bigger than this ?
        pointer_R_big = 1
        R_big[0, 0] = np.sqrt(1 + np.linalg.norm(P_S_0)**2)

        norms_P_S = [P_S_0.dot(P_S_0)]

        for t in range(1, T):
            # Step 1 of MNP
            #implement step 1(b) and obtain (Ji, Jk)
            z_t = self.compute_PS_dot_v(opt_lambda, P_S_small, P_S_index_list, z_star)

            max_norm_P_S = max(norms_P_S)
            
            Ji, Jk = self.lmo(z_t)
            P_Ji_Jk = self.build_vector_from_indices(Ji, Jk) - z_star
            
            #find max |P_J| for J \in active set
            if z_t.dot(P_Ji_Jk) > z_t.dot(z_t) - Z1 * max(P_Ji_Jk.dot(P_Ji_Jk), max_norm_P_S):
                if self.verbose:
                    print(f"Break at 1(c) with size active set = {len(active_set)}")
                break
            if (Ji, Jk) in active_set:
                if self.verbose:
                    print("Break at 1(d)")
                break
            else:
                
                active_set.append((Ji, Jk))
                is_index_added[Ji] += 1
                opt_lambda = np.concatenate((opt_lambda, [0])) 
                #update R
                r = sp.linalg.solve_triangular(R_big[:pointer_R_big, :pointer_R_big].T, 1 + self.compute_PS_T_dot_v(P_Ji_Jk, P_S_small, P_S_index_list, z_star), lower=True)
                #r = sp.linalg.solve_triangular(R.T, 1 + P_S.T @ P_Ji_Jk, lower=True)
                
                rho = np.sqrt(1 + P_Ji_Jk.dot(P_Ji_Jk) - r.dot(r))
                new_col = np.concatenate((r, [rho]))

                #update R_big
                if pointer_R_big == R_big.shape[1]:
                    #this should only happen if we have deleted many columns of R_big
                    if self.verbose:
                        print("Adding columns to R_big")
                    factor_new_col = 2
                    R_big_new = np.zeros((self.n+self.m, R_big.shape[1] * factor_new_col))
                    R_big_new[:, :R_big.shape[1]] = R_big
                    R_big = R_big_new
                R_big[:pointer_R_big+1, pointer_R_big] = new_col
                pointer_R_big += 1
                #update P_S
                P_S_small, P_S_index_list = self.update_PS(P_S_small, P_S_index_list, Ji, Jk, z_star)
                norms_P_S.append(P_Ji_Jk.dot(P_Ji_Jk))

            v = - np.zeros(1)
            counter = 0
            while np.any(v < Z2):
                counter += 1
                # Step 2 of MNP
                u_bar = sp.linalg.solve_triangular(R_big[:pointer_R_big, :pointer_R_big].T, np.ones(pointer_R_big), lower=True)
                u = sp.linalg.solve_triangular(R_big[:pointer_R_big, :pointer_R_big], u_bar, lower=False)
                v = u / np.sum(u)
                #Step 2(b)
                if np.all(v > Z2):
                    opt_lambda = v
                    break
                else:
                    pos = np.where(opt_lambda - v > Z3, opt_lambda/(opt_lambda - v), np.inf)
                    theta = min(1, np.min(pos))
                    #opt_lambda = theta * opt_lambda + (1-theta)*v
                    opt_lambda = ( 1- theta) * opt_lambda + theta*v
                    opt_lambda = np.where( opt_lambda < Z2, 0, opt_lambda)
                    indices_to_remove = np.where(opt_lambda==0)
                    if indices_to_remove[0].shape[0] > 1:
                        print("more than one index to remove, this might cause problems")
                    I = indices_to_remove[0][0]
                    is_index_added[active_set[I][0]] -= 1
                    
                    opt_lambda = np.delete(opt_lambda, I)
                    P_S_small = np.delete(P_S_small, I, axis=1)
                    del P_S_index_list[I]
                    del norms_P_S[I]
                    del active_set[I]

                    #update R
                    R_big = np.delete(R_big, I, axis=1)
                    pointer_R_big -= 1
                    while I < pointer_R_big:
                        a = R_big[I, I]
                        b = R_big[I+1, I]
                        c = np.sqrt(a**2 + b**2)
                        new_R_I = (a * R_big[I, :] + b * R_big[I+1, :]) / c
                        new_R_I1 = (-b * R_big[I, :] + a * R_big[I+1, :])/c
                        R_big[I, :] = new_R_I
                        R_big[I+1, :] = new_R_I1
                        I += 1
                    #at this point the last row of R should be 0
                        #print(R_big[pointer_R_big, :])
                    assert np.linalg.norm(R_big[pointer_R_big, :]) < 1e-6
                    
            nb_missing_indices = self.n - np.sum(is_index_added > 0)
            #if nb_missing_indices == 0:
            #    print(f"Stopping at iteration {t}/{T} with ||z_t||ˆ2 = {np.linalg.norm(z_t)**2} after all indices collected. size active set = {len(active_set)}")
            #    break
            if self.verbose:
                freq = int(self.n / 10)
                if (t-1) % freq == 0:
                    print(f"At iteration {t}/{T}, ||z_t||^2 = {np.linalg.norm(z_t)**2}, size active set = {len(active_set)}, nb_missing_indices = {self.n - np.sum(is_index_added > 0)}")
        return self.build_final_solution(opt_lambda=opt_lambda, active_set=active_set), self.n * z_t[:self.m] + self.n * z_star[:self.m]
    

    def build_final_solution(self, opt_lambda, active_set):
        y_dic_final = {}
        for i in range(self.n):
            matching_indices = [(index, j, k) for index, (j, k) in enumerate(active_set) if i == j]
            sum_lambda = sum([opt_lambda[int(index)] for (index, j, k) in matching_indices])
            y_dic_final[i] = [np.zeros((self.m, len(matching_indices))), np.zeros(len(matching_indices))]

            for l, (index, j, k) in enumerate(matching_indices):
                y_dic_final[i][0][:, l] = self.y_dic_integer_indexed[int(j)][:, int(k)]
                y_dic_final[i][1][l] = opt_lambda[index]/sum_lambda
        
        #now convert to the original indexing of the dictionary
        y_dic_final = self.convert_dict_to_original_indexing(y_dic_final)
        return CaratheodoryOutput(y_dic=y_dic_final)
    

class MetaCaratheodoryMNPSolve():
    def __init__(self,
                 y_dic,
                 weights_dic,
                 verbose:bool = True):
        self.y_dic = y_dic
        self.weights_dic = weights_dic
        self.y_dic_integer_indexed = self.compute_integer_indexed_y_dic()
        self.weights_dic_integer_indexed = self.compute_integer_indexed_weights_dic()
        self.conversion_dict = self.compute_conversion_dict() #to help 
        self.m = self.y_dic_integer_indexed[0].shape[0]  # Assuming all y_dic[i] have the same shape
        self.n = len(y_dic)
        self.verbose = verbose


    def compute_integer_indexed_y_dic(self):
        new_y_dic = {}
        for i, (key, pair) in enumerate(self.y_dic.items()):
            new_y_dic[i] = pair
        return new_y_dic
    
    def compute_integer_indexed_weights_dic(self):
        new_weights_dic = {}
        for i, (key, pair) in enumerate(self.weights_dic.items()):
            new_weights_dic[i] = pair
        return new_weights_dic

    def compute_conversion_dict(self):
        conversion_dict = {}
        for i, original_indexed in enumerate(self.y_dic):
            conversion_dict[i] = original_indexed
        return conversion_dict
    
    def convert_dict_to_original_indexing(self, dic):
        new_dic = {}
        #convert an integer-indexed dictionary to the original indexing
        for i, (key, pair) in enumerate(dic.items()):
            new_dic[self.conversion_dict[key]] = pair
        return new_dic

    def solve(self, z, T, nb_indices_considered, Z1=1e-12, Z2=1e-10, Z3=1e-10):


        nb_trivial_convex_combinations = 0
        nb_caratheodory_calls = 0
        final_y_dic = {}
        
        #create the instance of z_tilde
        z_tilde = z.copy()

        #get the first small y_dic
        small_y_dic = {}
        for i in range(nb_indices_considered):
            small_y_dic[i] = self.y_dic_integer_indexed[i]

        start_index = nb_indices_considered

        stop_condition = False
        while not stop_condition:
            
            #if end_index < self.n - np.floor(self.n/R):
            #    end_index = self.n #for the last block we just take everything that is left


            z_star = z_tilde.copy()
            for i in range(start_index, self.n):
                z_star -=  self.y_dic_integer_indexed[i] @ self.weights_dic_integer_indexed[i]

            carathodory_mnp_solver = CaratheodoryMNPSolve(y_dic=small_y_dic, verbose=False)
            caratheodory_output, _ = carathodory_mnp_solver.solve(z=z_star, T=T)
            nb_caratheodory_calls += 1

            #now update z_tilde
            keys_to_remove = []
            small_y_dic_output = caratheodory_output.y_dic
            for key in small_y_dic_output:
                if small_y_dic_output[key][1].shape[0] == 1:
                    #here we have found a trivial convex combination, so we remove the corresponding element from z_tilde
                    z_tilde -= small_y_dic_output[key][0][:, 0]
                    final_y_dic[key] = small_y_dic_output[key][0][:, 0]
                    keys_to_remove.append(key)
                    nb_trivial_convex_combinations += 1
            for key in keys_to_remove:
                small_y_dic.pop(key, None)

            #now increase the size of small_y_dic
            if start_index < self.n:
                while len(small_y_dic) < nb_indices_considered:
                    small_y_dic[start_index] = self.y_dic_integer_indexed[start_index]
                    start_index += 1
                    if start_index >= self.n:
                        break
            else:
                #in that case there is nothing to add to y_dic and we are done
                stop_condition = True
            
            print(f"After {nb_caratheodory_calls} Caratheodory algorithms, {nb_trivial_convex_combinations} trivial convex combinations found.")
        
        
        #build the final dictionary in the required format
        #start with the keys already there, i.e. the ones corresponding to trivial convex combinations
        for key in final_y_dic:
            trivial_cvx_vector = final_y_dic[key].copy()
            final_y_dic[key] = []
            final_y_dic[key].append(np.zeros((self.m, 1)))
            final_y_dic[key][0][:, 0] = trivial_cvx_vector
            final_y_dic[key].append(np.ones(1))
        #now add the missing keys, i.e. the ones corresponding to non trivial convex combinations
        for key in small_y_dic:
            assert key not in final_y_dic
            final_y_dic[key] = []
            final_y_dic[key].append(small_y_dic_output[key][0])
            final_y_dic[key].append(small_y_dic_output[key][1])

        
        final_y_dic = self.convert_dict_to_original_indexing(final_y_dic)
        return CaratheodoryOutput(y_dic=final_y_dic)