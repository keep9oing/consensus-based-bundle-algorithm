import numpy as np
import copy


class CBBA_agent():
  def __init__(self, id = None, task_num = None, agent_num = None, L_t = None):

    self.task_num = task_num
    self.agent_num = agent_num

    # Agent ID
    self.id = id

    # Local Winning Agent List
    self.z = np.ones(self.task_num, dtype=np.int8) * self.id
    # Local Winning Bid List
    self.y = np.array([ 0 for _ in range(self.task_num)], dtype=np.float64)
    # Bundle
    self.b = []
    # Path
    self.p = []
    # Maximum Task Number
    self.L_t = L_t
    # Local Clock
    self.time_step = 0
    # Time Stamp List
    self.s = {a:self.time_step for a in range(self.agent_num)}

    # This part can be modified depend on the problem
    self.state = np.random.uniform(low=0, high=1, size=(1,2)) # Agent State (Position)
    self.c = np.zeros(self.task_num) # Initial Score (Euclidean Distance)

  def set_state(self, state):
    """
    Set state of agent
    """
    self.state = state

  def send_message(self):
    """
    Return local winning bid list
    [output]
    y: winning bid list (list:task_num)
    z: winning agent list (list:task_num)
    s: Time Stamp List (Dict:{agent_id:update_time})
    """
    return self.y.tolist(), self.z.tolist(), self.s

  def receive_message(self, Y):
    self.Y = Y

  def build_bundle(self, task):
    """
    Construct bundle and path list with local information
    """
    J = [j for j in range(self.task_num)]

    while len(self.b) < self.L_t:
      # Calculate S_p for constructed path list
      S_p = 0
      if len(self.p) > 0:
        distance_j = 0
        distance_j += np.linalg.norm(self.state.squeeze()-task[self.p[0]])
        S_p += np.exp(-distance_j)
        for p_idx in range(len(self.p)-1):
          distance_j += np.linalg.norm(task[self.p[p_idx]]-task[self.p[p_idx+1]])
          S_p += np.exp(-distance_j)

      # Calculate c_ij for each task j
      best_pos = {}
      for j in J:
        c_list = []
        if j in self.b: # If already in bundle list
          self.c[j] = 0 # Minimum Score
        else:
          for n in range(len(self.p)+1):
            p_temp = copy.deepcopy(self.p)
            p_temp.insert(n,j)
            c_temp = 0
            distance_j = 0
            distance_j += np.linalg.norm(self.state.squeeze()-task[p_temp[0]])
            c_temp += np.exp(-distance_j)
            if len(p_temp) > 1:
              for p_loc in range(len(p_temp)-1):
                distance_j += np.linalg.norm(task[p_temp[p_loc]]-task[p_temp[p_loc+1]])
                c_temp += np.exp(-distance_j)

            c_jn = c_temp-S_p
            c_list.append(c_jn)

          max_idx = np.argmax(c_list)
          c_j = c_list[max_idx]
          self.c[j] = c_j
          best_pos[j] = max_idx

      h = (self.c > self.y)
      if sum(h)==0:# No valid task
        break
      self.c[~h] = 0
      J_i = np.argmax(self.c)
      n_J = best_pos[J_i]

      self.b.append(J_i)
      self.p.insert(n_J,J_i)

      self.y[J_i] = self.c[J_i]
      self.z[J_i] = self.id


  def update_task(self):
    """
    [input]
    Y: winning bid lists from neighbors (dict:{neighbor_id:(winning bid_list, winning agent list, time stamp list)})
    time: for simulation,
    """

    old_p = copy.deepcopy(self.p)

    id_list = list(self.Y.keys())
    id_list.insert(0, self.id)

    # Update time list
    for id in list(self.s.keys()):
      if id in id_list:
        self.s[id] = self.time_step
      else:
        s_list = []
        for neighbor_id in id_list[1:]:
          s_list.append(self.Y[neighbor_id][2][id])
        if len(s_list) > 0:
          self.s[id] = max(s_list)

    ## Update Process
    for j in range(self.task_num):
      for k in id_list[1:]:
        y_k = self.Y[k][0]
        z_k = self.Y[k][1]
        s_k = self.Y[k][2]

        z_ij = self.z[j]
        z_kj = z_k[j]
        y_kj = y_k[j]

        i = self.id
        y_ij = self.y[j]

        ## Rule Based Update
        # Rule 1~4
        if z_kj == k:
          # Rule 1
          if z_ij == self.id:
            if y_kj > y_ij:
              self.__update(j,y_kj,z_kj)
            elif abs(y_kj - y_ij) < np.finfo(float).eps: # Tie Breaker
              if k < self.id:
                self.__update(j,y_kj,z_kj)
            else:
              self.__leave()
          # Rule 2
          elif z_ij == k:
            self.__update(j,y_kj,z_kj)
          # Rule 3
          elif z_ij != -1:
            m = z_ij
            if (s_k[m] > self.s[m]) or (y_kj > y_ij):
              self.__update(j,y_kj,z_kj)
            elif abs(y_kj-y_ij) < np.finfo(float).eps: # Tie Breaker
              if k < self.id:
                self.__update(j,y_kj,z_kj)
          # Rule 4
          elif z_ij == -1:
            self.__update(j,y_kj,z_kj)
          else:
            raise Exception("Error while updating")
        # Rule 5~8
        elif z_kj == i:
          # Rule 5
          if z_ij == i:
            self.__leave()
          # Rule 6
          elif z_ij == k:
            self.__reset(j)
          # Rule 7
          elif z_ij != -1:
            m = z_ij
            if s_k[m] > self.s[m]:
              self.__reset(j)
          # Rule 8
          elif z_ij == -1:
            self.__leave()
          else:
            raise Exception("Error while updating")
        # Rule 9~13
        elif z_kj != -1:
          m = z_kj
          # Rule 9
          if z_ij == i:
            if (s_k[m]>=self.s[m]) and (y_kj > y_ij):
              self.__update(j,y_kj,z_kj)
            elif (s_k[m]>=self.s[m]) and (abs(y_kj-y_ij) < np.finfo(float).eps): # Tie Breaker
              if m < self.id:
                self.__update(j,y_kj,z_kj)
          # Rule 10
          elif z_ij == k:
            if (s_k[m]>self.s[m]):
              self.__update(j,y_kj,z_kj)
            else:
              self.__reset(j)
          # Rule 11
          elif z_ij == m:
            if (s_k[m] > self.s[m]):
              self.__update(j,y_kj,z_kj)
          # Rule 12
          elif z_ij != -1:
            n = z_ij
            if (s_k[m] > self.s[m]) and (s_k[n] > self.s[n]):
              self.__update(j,y_kj,z_kj)
            elif (s_k[m] > self.s[m]) and (y_kj > y_ij):
              self.__update(j,y_kj,z_kj)
            elif (s_k[m]>self.s[m]) and (abs(y_kj-y_ij) < np.finfo(float).eps): # Tie Breaker
              if m < n:
                self.__update(j,y_kj,z_kj)
            elif (s_k[n]>self.s[n]) and (self.s[m]>s_k[m]):
              self.__update(j,y_kj,z_kj)
          # Rule 13
          elif z_ij == -1:
            if (s_k[m] > self.s[m]):
              self.__update(j,y_kj,z_kj)
          else:
            raise Exception("Error while updating")
        # Rule 14~17
        elif z_kj == -1:
          # Rule 14
          if z_ij == i:
            self.__leave()
          # Rule 15
          elif z_ij == k:
            self.__update(j,y_kj,z_kj)
          # Rule 16
          elif z_ij != -1:
            m = z_ij
            if s_k[m] > self.s[m]:
              self.__update(j,y_kj,z_kj)
          # Rule 17
          elif z_ij == -1:
            self.__leave()
          else:
            raise Exception("Error while updating")
        else:
          raise Exception("Error while updating")

    n_bar = len(self.b)
    # Get n_bar
    for n in range(len(self.b)):
      b_n = self.b[n]
      if self.z[b_n] != self.id:
        n_bar = n
        break

    b_idx1 = copy.deepcopy(self.b[n_bar+1:])

    if len(b_idx1) > 0:
      self.y[b_idx1] = 0
      self.z[b_idx1] = -1

    if n_bar < len(self.b):
      del self.b[n_bar:]

    self.p = []
    for task in self.b:
      self.p.append(task)

    self.time_step += 1

    converged = False
    if old_p == self.p:
      converged = True

    return converged


  def __update(self, j, y_kj, z_kj):
    """
    Update values
    """
    self.y[j] = y_kj
    self.z[j] = z_kj

  def __reset(self, j):
    """
    Reset values
    """
    self.y[j] = 0
    self.z[j] = -1 # -1 means "none"

  def __leave(self):
    """
    Do nothing
    """
    pass

if __name__=="__main__":
  import matplotlib.pyplot as plt

  np.random.seed(2)

  task_num = 3
  robot_num = 3

  task = np.random.uniform(low=0,high=1,size=(task_num,2))
  # task = np.array([[0,1],[1,1],[1,2]])

  robot_list = [CBBA_agent(id=i, task_num=task_num, agent_num=robot_num, L_t=task.shape[0]) for i in range(robot_num)]
  # robot_list[0].state = np.array([[0,0]])
  # robot_list[1].state = np.array([[1,0]])

  # Network Initialize
  G = np.ones((robot_num, robot_num)) # Fully connected network
  # G[0,1]=0
  # G[1,0]=0

  t = 0 # Iteration number


  while True:
    converged_list = [] # Converged List

    print("==Iteration {}==".format(t))
    ## Phase 1: Auction Process
    print("Auction Process")
    for robot in robot_list:
      # select task by local information
      robot.build_bundle(task)

    print("Bundle")
    for robot in robot_list:
      print(robot.b)
    print("Path")
    for robot in robot_list:
      print(robot.p)

    ## Communication stage
    print("Communicating...")
    # Send winning bid list to neighbors (depend on env)
    message_pool = [robot.send_message() for robot in robot_list]

    for robot_id, robot in enumerate(robot_list):
      # Recieve winning bidlist from neighbors
      g = G[robot_id]

      connected, = np.where(g==1)
      connected = list(connected)
      connected.remove(robot_id)

      if len(connected) > 0:
        Y = {neighbor_id:message_pool[neighbor_id] for neighbor_id in connected}
      else:
        Y = None

      robot.receive_message(Y)

    ## Phase 2: Consensus Process
    print("Consensus Process")
    for robot in robot_list:
      # Update local information and decision
      if Y is not None:
        converged = robot.update_task()
        converged_list.append(converged)

    print("Bundle")
    for robot in robot_list:
      print(robot.b)
    print("Path")
    for robot in robot_list:
      print(robot.p)

    t += 1

    if sum(converged_list) == robot_num:
      break

print("Finished")


