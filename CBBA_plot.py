from CBBA import CBBA_agent

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# For build GIF
import imageio
import os

np.random.seed(3)

task_num = 20
robot_num = 4

task = np.random.uniform(low=0,high=1,size=(task_num,2))

robot_list = [CBBA_agent(id=i, task_num=task_num, agent_num=robot_num, L_t=task.shape[0]) for i in range(robot_num)]

# Network Initialize
G = np.ones((robot_num, robot_num)) # Fully connected network
# disconnect link arbitrary
G[2,3]=0
G[3,2]=0
G[1,2]=0
G[2,1]=0
G[1,3]=0
G[3,1]=0

fig, ax = plt.subplots()
ax.set_xlim((-0.1,1.1))
ax.set_ylim((-0.1,1.1))

ax.plot(task[:,0],task[:,1],'rx',label="Task")
robot_pos = np.array([r.state[0].tolist() for r in robot_list])
ax.plot(robot_pos[:,0],robot_pos[:,1],'b^',label="Robot")

for i in range(robot_num-1):
  for j in range(i+1,robot_num):
    if G[i][j] == 1:
      ax.plot([robot_pos[i][0],robot_pos[j][0]],[robot_pos[i][1],robot_pos[j][1]],'g--',linewidth=1)

handles, labels = ax.get_legend_handles_labels()
custom_line = Line2D([0], [0], color="g",linestyle="--",label="communication")
handles.append(custom_line)
ax.legend(handles=handles)

t = 0 # Iteration number
assign_plots = []
max_t = 100
plot_gap = 0.1

save_gif = False
filenames = []

if save_gif:
  if not os.path.exists("my_gif"):
    os.makedirs("my_gif")

while True:
  converged_list = []

  print("==Iteration {}==".format(t))
  ## Phase 1: Auction Process
  print("Auction Process")
  for robot_id, robot in enumerate(robot_list):
    # select task by local information
    robot.build_bundle(task)

    ## Plot
    if len(robot.p) > 0:
      x_data=[robot.state[0][0]]+task[robot.p,0].tolist()
      y_data=[robot.state[0][1]]+task[robot.p,1].tolist()
    else:
      x_data=[robot.state[0][0]]
      y_data=[robot.state[0][1]]
    if t == 0:
      assign_line, = ax.plot(x_data,y_data,'k-',linewidth=1)
      assign_plots.append(assign_line)
    else:
      assign_plots[robot_id].set_data(x_data,y_data)

  print("Bundle")
  for robot in robot_list:
    print(robot.b)
  print("Path")
  for robot in robot_list:
    print(robot.p)

  ## Plot
  ax.set_title("Time Step:{}, Bundle Construct".format(t))
  plt.pause(plot_gap)
  if save_gif:
    filename = f'{t}_B.png'
    filenames.append(filename)
    plt.savefig(filename)

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
  for robot_id, robot in enumerate(robot_list):
    # Update local information and decision
    if Y is not None:
      converged = robot.update_task()
      converged_list.append(converged)

    ## Plot
    if len(robot.p) > 0:
      x_data=[robot.state[0][0]]+task[robot.p,0].tolist()
      y_data=[robot.state[0][1]]+task[robot.p,1].tolist()
    else:
      x_data=[robot.state[0][0]]
      y_data=[robot.state[0][1]]

    assign_plots[robot_id].set_data(x_data,y_data)

  ## Plot
  ax.set_title("Time Step:{}, Consensus".format(t))
  plt.pause(plot_gap)
  if save_gif:
    filename = f'./my_gif/{t}_C.png'
    filenames.append(filename)
    plt.savefig(filename)

  print("Bundle")
  for robot in robot_list:
    print(robot.b)
  print("Path")
  for robot in robot_list:
    print(robot.p)

  t += 1

  if sum(converged_list) == robot_num:
    ax.set_title("Time Step:{}, Converged!".format(t))
    break
  if t>max_t:
    ax.set_title("Time Step:{}, Max time step overed".format(t))
    break


if save_gif:
    filename = f'./my_gif/{t}_F.png'
    filenames.append(filename)
    plt.savefig(filename)

    #build gif
    files=[]
    for filename in filenames:
        image = imageio.imread(filename)
        files.append(image)
    imageio.mimsave("./my_gif/mygif.gif", files, format='GIF', fps = 0.5)
    with imageio.get_writer('./my_gif/mygif.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    # Remove files
    for filename in set(filenames):
        os.remove(filename)

plt.show()
