import numpy as np
import copy
from scipy.spatial import distance_matrix


class CBAA_agent():
  def __init__(self, id=0, task=None):
    """
    c: individual score list
    x: local assignment list
    y: local winning bid list
    state: state of the robot
    """

    self.task_num = task.shape[0]

    # Local Task Assignment List
    self.x = np.zeros(self.task_num, dtype=np.int8)
    # Local Winning Bid List
    self.y = np.array([ -np.inf for _ in range(self.task_num)])


    # This part can be modified depend on the problem
    self.state = np.random.uniform(low=0, high=1, size=(1,2)) # Agent State (Position)
    self.c = -distance_matrix(self.state,task).squeeze() # Score (Euclidean Distance)

    # Agent ID
    self.id = id

  def select_task(self):
    if sum(self.x) == 0:
      # Valid Task List
      h = (self.c > self.y)
      if h.any():
        # Just for euclidean distance score (negative)
        c = copy.deepcopy(self.c)
        c[h==False] = -np.inf

        self.J = np.argmax(c)
        self.x[self.J] = 1
        self.y[self.J] = self.c[self.J]

  def update_task(self, Y=None):
    """
    [input]
    Y: winning bid lists from neighbors (dict:{neighbor_id:bid_list})
    """

    id_list = list(Y.keys())
    id_list.insert(0, self.id)

    y_list = np.array(list(Y.values()))

    ## Update local winning bid list
    # When recive only one message
    if len(y_list.shape)==1:
      # make shape as (1,task_num)
      y_list = y_list[None,:]

    # Append the agent's local winning bid list and neighbors'
    y_list = np.vstack((self.y[None,:],y_list))

    self.y = y_list.max(0)

    ## Outbid check
    # Winner w.r.t the updated local winning bid list
    max_id = np.argmax(y_list[:,self.J])
    z = id_list[max_id]
    # If the agent is not the winner
    if z != self.id:
        # Release the assignment
        self.x[self.J] = 0

  def send_message(self):
    """
    Return local winning bid list
    [output]
    y: winning bid list (list:task_num)
    """
    return self.y.tolist()

if __name__=="__main__":

  task_num = 5
  robot_num = 5

  task = np.random.uniform(low=0,high=1,size=(task_num,2))

  robot_list = [CBAA_agent(id=i, task=task) for i in range(robot_num)]

  # Network Initialize
  G = np.ones((robot_num, robot_num)) # Fully connected network
  # G[0,1]=0
  # G[1,0]=0

  t = 0 # Iteration number
  while True:
    print("==Iteration {}==".format(t))
    ## Phase 1: Auction Process
    print("Auction Process")
    for robot in robot_list:
      # select task by local information
      robot.select_task()
      print(robot.x)

    ## Phase 2: Consensus Process
    print("Consensus Process")
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

      # Update local information and decision
      if Y is not None:
        robot.update_task(Y)

      print(robot.x)

    t += 1


