import latentsafesets.envs.simple_point_bot as spb

import numpy as np
from tqdm import tqdm


def evaluate_safe_set(s_set,
                      env,
                      file=None,
                      plot=True,
                      show=False,
                      skip=2):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            state = env._state_to_image((x, y)) / 255##this is for global coordinates!
            #state = env._state_to_image_relative((x, y)) / 255##this is for ego coordinates!
            row_states.append(state)
        vals = s_set.safe_set_probability_np(np.array(row_states)).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y+1, ::2], data[y+1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show)

    return data


def evaluate_value_func(value_func,
                        env,
                        file=None,
                        plot=True,
                        show=False,
                        skip=2):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            state = env._state_to_image((x, y)) / 255#this is for global coordinates!
            #state = env._state_to_image_relative((x, y)) / 255#this is for ego coordinates!
            row_states.append(state)
        vals = value_func.get_value_np(np.array(row_states)).squeeze()
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show)

    return data


def evaluate_constraint_func(constraint,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=2):
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            state = env._state_to_image((x, y)) / 255##this is for global coordinates!
            #state = env._state_to_image_relative((x, y)) / 255##this is for ego coordinates!
            row_states.append(state)
        vals = constraint.prob(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdot_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0)):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    #if walls is None:
    xmove = 0  # -25#30#
    ymove = 0  #-45#-40#-35#-33#-30#-25#
    lux = 50#75#105#
    luy = 55#40#
    width = 25#20#
    height = 40#50#
    walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))] #[((75, 55), (100, 95))] #the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        row_statesc = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            #old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            old_state = env._state_to_image((x, y)) / 255##this is for global coordinates!
            row_states.append(old_state)
            #print('old_state',old_state)#tuple

            #print('selfwall_coords[0][0]',selfwall_coords[0][0])#tuple
            #print('(old_state <= selfwall_coords[0][0])',(old_state <= selfwall_coords[0][0]))
            if (old_state <= selfwall_coords[0][0]).all():  # old_state#check it!
                reldistold = old_state - selfwall_coords[0][0]  # np.linalg.norm()
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] <= \
                    selfwall_coords[0][0][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][0][1]])
            elif old_state[0] >= selfwall_coords[0][1][0] and old_state[1] <= selfwall_coords[0][0][1]:
                reldistold = old_state - (selfwall_coords[0][1][0], selfwall_coords[0][0][1])
            elif old_state[0] >= selfwall_coords[0][1][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][1][0], 0])
            elif (old_state >= selfwall_coords[0][1]).all():  # old_state
                reldistold = old_state - selfwall_coords[0][1]
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] >= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][1][1]])
            elif old_state[0] <= selfwall_coords[0][0][0] and old_state[1] >= selfwall_coords[0][1][1]:
                reldistold = (old_state - (selfwall_coords[0][0][0], selfwall_coords[0][1][1]))
            elif old_state[0] <= selfwall_coords[0][0][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][0][0], 0])
            else:
                # print(old_state)#it can be [98.01472841 92.11425524]
                reldistold = np.array([0, 0])  # 9.9#

            rda=np.concatenate((reldistold,action))#thanks it is one-by-one
            row_states.append(rda)
            #print('obs.shape',obs.shape)
            #print('action',action)
            #rdal=np.concatenate((obs,action))
            #row_states.append(rdal)

        vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states),already_embedded = True).squeeze()  # it is like calling forward of const_estimator!

        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data


def evaluate_cbfdotc_func(cbfdot,#c means circle
                         env,
                         file=None,
                         plot=True,
                         show=False,
                         skip=1,
                         action=(0, 0)):  # (1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    # if walls is None:
    xmove = 0  # -25#30#
    ymove = -45  # -40#-35#-33#-30#-25#0  #
    lux = 50
    luy = 55
    width = 20  # 25#
    height = 50  # 40#
    walls = [((lux + xmove, luy + ymove), (
    lux + width + xmove, luy + height + ymove))]  # [((75, 55), (100, 95))] #the position and dimension of the wall
    # self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        row_statesc = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x, y))  # env._state_to_image((x, y)) / 255
            # print('old_state',old_state)#tuple
            # print('selfwall_coords[0][0]',selfwall_coords[0][0])#tuple
            # print('(old_state <= selfwall_coords[0][0])',(old_state <= selfwall_coords[0][0]))
            if (old_state <= selfwall_coords[0][0]).all():  # old_state#check it!
                reldistold = old_state - selfwall_coords[0][0]  # np.linalg.norm()
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] <= \
                    selfwall_coords[0][0][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][0][1]])
            elif old_state[0] >= selfwall_coords[0][1][0] and old_state[1] <= selfwall_coords[0][0][1]:
                reldistold = old_state - (selfwall_coords[0][1][0], selfwall_coords[0][0][1])
            elif old_state[0] >= selfwall_coords[0][1][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][1][0], 0])
            elif (old_state >= selfwall_coords[0][1]).all():  # old_state
                reldistold = old_state - selfwall_coords[0][1]
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] >= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][1][1]])
            elif old_state[0] <= selfwall_coords[0][0][0] and old_state[1] >= selfwall_coords[0][1][1]:
                reldistold = (old_state - (selfwall_coords[0][0][0], selfwall_coords[0][1][1]))
            elif old_state[0] <= selfwall_coords[0][0][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][0][0], 0])
            else:
                # print(old_state)#it can be [98.01472841 92.11425524]
                reldistold = np.array([0, 0])  # 9.9#
            rda = np.concatenate((reldistold, action))  # thanks it is one-by-one
            row_states.append(rda)

            # the circle part
            # device = se.device
            centerx = 115  # 118#
            centery = 85  # 80#75#
            circlecenter = np.array([centerx, centery])  # circlecenter = torch.tensor([centerx, centery]).to(device)
            circleradius = 20  # 19#15#14#30#25#
            #print('old_state',old_state)
            #print(circlecenter,circlecenter)
            rdc = old_state - circlecenter  # torch.asarray(circlecenter)  # relative distance vector#print('rd',rd)
            rdtan2c = np.arctan2(rdc[1], rdc[0])  # torch.atan2(rdc[:, :, 1], rdc[:, :, 0])  # get the angle
            rdyc = circleradius * np.sin(rdtan2c)  # relative distance in y direction led by the circular obstacle
            rdxc = circleradius * np.cos(rdtan2c)
            rdrc = np.array([rdxc,rdyc])#np.concatenate(
                #(rdxc.reshape(rdxc.shape[0], rdxc.shape[1], 1), rdyc.reshape(rdyc.shape[0], rdyc.shape[1], 1)),
                #dim=2)  # dim: (1000,5,2)#rdr means relative distance induced by the radius of the circular obstacle
            # print('rdr',rdr)
            rd8c = rdc - rdrc  # print('rd8',rd8)
            rdac = np.concatenate((rd8c, action))  # thanks it is one-by-one
            row_statesc.append(rdac)

        vals = cbfdot.cbfdots(np.array(row_states)).squeeze()  # it is like calling forward of const_estimator!
        valsc = cbfdot.cbfdots(np.array(row_statesc)).squeeze()  # it is like calling forward of const_estimator!
        vals=vals+valsc
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdotconly_func(cbfdot,#c means circle
                         env,
                         file=None,
                         plot=True,
                         show=False,
                         skip=1,
                         action=(0, 0)):  # (1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        #row_states = []
        row_statesc = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x, y))  # env._state_to_image((x, y)) / 255
            # the circle part
            # device = se.device
            centerx = 115  # 118#
            centery = 85  # 80#75#
            circlecenter = np.array([centerx, centery])  # circlecenter = torch.tensor([centerx, centery]).to(device)
            circleradius = 20  # 19#15#14#30#25#
            #print('old_state',old_state)
            #print(circlecenter,circlecenter)
            rdc = old_state - circlecenter  # torch.asarray(circlecenter)  # relative distance vector#print('rd',rd)
            rdtan2c = np.arctan2(rdc[1], rdc[0])  # torch.atan2(rdc[:, :, 1], rdc[:, :, 0])  # get the angle
            rdyc = circleradius * np.sin(rdtan2c)  # relative distance in y direction led by the circular obstacle
            rdxc = circleradius * np.cos(rdtan2c)
            rdrc = np.array([rdxc,rdyc])#np.concatenate(
                #(rdxc.reshape(rdxc.shape[0], rdxc.shape[1], 1), rdyc.reshape(rdyc.shape[0], rdyc.shape[1], 1)),
                #dim=2)  # dim: (1000,5,2)#rdr means relative distance induced by the radius of the circular obstacle
            # print('rdr',rdr)
            rd8c = rdc - rdrc  # print('rd8',rd8)
            rdac = np.concatenate((rd8c, action))  # thanks it is one-by-one
            row_statesc.append(rdac)

        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze()  # it is like calling forward of const_estimator!
        valsc = cbfdot.cbfdots(np.array(row_statesc)).squeeze()  # it is like calling forward of const_estimator!
        vals=valsc#vals+valsc
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdotlatent_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0)):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))#this is for plan a that takes latent state and then get cbf values
    #if walls is None:
    xmove = 0  # -25#30#
    ymove = 0  #-45#-40#-35#-33#-30#-25#
    lux = 50#75#105#
    luy = 55#40#
    width = 25#20#
    height = 40#50#
    walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))] #[((75, 55), (100, 95))] #the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        row_statesc = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            #old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            old_state = env._state_to_image((x, y)) / 255#can only be used for global coordinates
            row_states.append(old_state)
            #print('old_state',old_state)#tuple

            #rda=np.concatenate((reldistold,action))#thanks it is one-by-one
            #row_states.append(rda)
            #print('obs.shape',obs.shape)
            #print('action',action)
            #rdal=np.concatenate((obs,action))
            #row_states.append(rdal)

        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states),already_embedded = True).squeeze()  # it is like calling forward of const_estimator!
        vals = cbfdot.cbfdots(np.array(row_states)).squeeze() #latent and latent plan a are the same in this case # it is like calling forward of const_estimator!
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdotlatentunbiased_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0)):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    #if walls is None:
    xmove = 0  # -25#30#
    ymove = 0  #-45#-40#-35#-33#-30#-25#
    lux = 50#75#105#
    luy = 55#40#
    width = 25#20#
    height = 40#50#
    walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))] #[((75, 55), (100, 95))] #the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        row_statesu = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            old_stateimage = env._state_to_image((x, y)) / 255
            #old_stateimage = env._state_to_image_relative((x, y)) / 255
            row_states.append(old_stateimage)
            #print('old_state',old_state)#tuple

            #print('selfwall_coords[0][0]',selfwall_coords[0][0])#tuple
            #print('(old_state <= selfwall_coords[0][0])',(old_state <= selfwall_coords[0][0]))
            if (old_state <= selfwall_coords[0][0]).all():  # old_state#check it!
                reldistold = old_state - selfwall_coords[0][0]  # np.linalg.norm()
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] <= \
                    selfwall_coords[0][0][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][0][1]])
            elif old_state[0] >= selfwall_coords[0][1][0] and old_state[1] <= selfwall_coords[0][0][1]:
                reldistold = old_state - (selfwall_coords[0][1][0], selfwall_coords[0][0][1])
            elif old_state[0] >= selfwall_coords[0][1][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][1][0], 0])
            elif (old_state >= selfwall_coords[0][1]).all():  # old_state
                reldistold = old_state - selfwall_coords[0][1]
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] >= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][1][1]])
            elif old_state[0] <= selfwall_coords[0][0][0] and old_state[1] >= selfwall_coords[0][1][1]:
                reldistold = (old_state - (selfwall_coords[0][0][0], selfwall_coords[0][1][1]))
            elif old_state[0] <= selfwall_coords[0][0][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][0][0], 0])
            else:
                # print(old_state)#it can be [98.01472841 92.11425524]
                reldistold = np.array([0, 0])  # 9.9#
            row_statesu.append(reldistold[0]**2+reldistold[1]**2-15**2)
            #row_statesu.append(reldistold[0]**2+reldistold[1]**2-5**2)
            #rda=np.concatenate((reldistold,action))#thanks it is one-by-one
            #row_states.append(rda)
            #print('obs.shape',obs.shape)
            #print('action',action)
            #rdal=np.concatenate((obs,action))
            #row_states.append(rdal)

        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states),already_embedded = True).squeeze()  # it is like calling forward of const_estimator!
        vals = cbfdot.cbfdots(np.array(row_states)).squeeze() #latent and latent plan a are the same in this case # it is like calling forward of const_estimator!
        #print('vals',vals)
        valsu=np.array(row_statesu).squeeze()
        #print('valsu', valsu)
        vals=vals-valsu
        #print('valsnew', vals)
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdotlatentgroundtruth_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0)):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    #if walls is None:
    xmove = 0  # -25#30#
    ymove = 0  #-45#-40#-35#-33#-30#-25#
    lux = 50#75#105#
    luy = 55#40#
    width = 25#20#
    height = 40#50#
    walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))] #[((75, 55), (100, 95))] #the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        row_statesu = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            old_stateimage = env._state_to_image((x, y)) / 255
            #old_stateimage = env._state_to_image_relative((x, y)) / 255
            row_states.append(old_stateimage)
            #print('old_state',old_state)#tuple

            #print('selfwall_coords[0][0]',selfwall_coords[0][0])#tuple
            #print('(old_state <= selfwall_coords[0][0])',(old_state <= selfwall_coords[0][0]))
            if (old_state <= selfwall_coords[0][0]).all():  # old_state#check it!
                reldistold = old_state - selfwall_coords[0][0]  # np.linalg.norm()
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] <= \
                    selfwall_coords[0][0][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][0][1]])
            elif old_state[0] >= selfwall_coords[0][1][0] and old_state[1] <= selfwall_coords[0][0][1]:
                reldistold = old_state - (selfwall_coords[0][1][0], selfwall_coords[0][0][1])
            elif old_state[0] >= selfwall_coords[0][1][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][1][0], 0])
            elif (old_state >= selfwall_coords[0][1]).all():  # old_state
                reldistold = old_state - selfwall_coords[0][1]
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] >= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][1][1]])
            elif old_state[0] <= selfwall_coords[0][0][0] and old_state[1] >= selfwall_coords[0][1][1]:
                reldistold = (old_state - (selfwall_coords[0][0][0], selfwall_coords[0][1][1]))
            elif old_state[0] <= selfwall_coords[0][0][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][0][0], 0])
            else:
                # print(old_state)#it can be [98.01472841 92.11425524]
                reldistold = np.array([0, 0])  # 9.9#
            row_statesu.append(reldistold[0]**2+reldistold[1]**2-15**2)
            #row_statesu.append(reldistold[0]**2+reldistold[1]**2-5**2)
            #rda=np.concatenate((reldistold,action))#thanks it is one-by-one
            #row_states.append(rda)
            #print('obs.shape',obs.shape)
            #print('action',action)
            #rdal=np.concatenate((obs,action))
            #row_states.append(rdal)

        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states),already_embedded = True).squeeze()  # it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze() #latent and latent plan a are the same in this case # it is like calling forward of const_estimator!
        #print('vals',vals)
        valsu=np.array(row_statesu).squeeze()
        #print('valsu', valsu)
        vals=valsu#vals-valsu
        #print('valsnew', vals)
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdotlatentunbiased13_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0),coeff=1/3):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    #if walls is None:
    xmove = 0  # -25#30#
    ymove = 0  #-45#-40#-35#-33#-30#-25#
    lux = 50#75#105#
    luy = 55#40#
    width = 25#20#
    height = 40#50#
    walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))] #[((75, 55), (100, 95))] #the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        row_statesu = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            old_stateimage = env._state_to_image((x, y)) / 255#, this is for using the global coordinate
            #old_stateimage = env._state_to_image_relative((x, y)) / 255#, this is for using the ego coordinate
            row_states.append(old_stateimage)
            if (old_state <= selfwall_coords[0][0]).all():  # old_state#check it!
                reldistold = old_state - selfwall_coords[0][0]  # np.linalg.norm()
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] <= \
                    selfwall_coords[0][0][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][0][1]])
            elif old_state[0] >= selfwall_coords[0][1][0] and old_state[1] <= selfwall_coords[0][0][1]:
                reldistold = old_state - (selfwall_coords[0][1][0], selfwall_coords[0][0][1])
            elif old_state[0] >= selfwall_coords[0][1][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][1][0], 0])
            elif (old_state >= selfwall_coords[0][1]).all():  # old_state
                reldistold = old_state - selfwall_coords[0][1]
            elif selfwall_coords[0][0][0] <= old_state[0] <= selfwall_coords[0][1][0] and old_state[1] >= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([0, old_state[1] - selfwall_coords[0][1][1]])
            elif old_state[0] <= selfwall_coords[0][0][0] and old_state[1] >= selfwall_coords[0][1][1]:
                reldistold = (old_state - (selfwall_coords[0][0][0], selfwall_coords[0][1][1]))
            elif old_state[0] <= selfwall_coords[0][0][0] and selfwall_coords[0][0][1] <= old_state[1] <= \
                    selfwall_coords[0][1][1]:
                reldistold = np.array([old_state[0] - selfwall_coords[0][0][0], 0])
            else:
                # print(old_state)#it can be [98.01472841 92.11425524]
                reldistold = np.array([0, 0])  # 9.9#
            row_statesu.append(reldistold[0]**2+reldistold[1]**2-15**2)
            #row_statesu.append(reldistold[0]**2+reldistold[1]**2-5**2)

        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states),already_embedded = True).squeeze()  # it is like calling forward of const_estimator!
        vals = cbfdot.cbfdots(np.array(row_states)).squeeze() #latent and latent plan a are the same in this case # it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots_planb(np.array(row_states)).squeeze()  #
        #vals = (vals) ** 3 #this is for plan b only!!! not for plan a!!! # -valsu#print('valsnew', vals)
        #print('vals',vals)
        valsu=np.array(row_statesu).squeeze()#print('valsu', valsu)
        vals=vals-valsu#print('valsnew', vals)
        #print('vals1',vals)
        valsmask=np.where(vals<=0,-1,1)
        vals=(np.abs(vals))**(coeff)#np.power(vals,1/3,dtype=complex)
        #print('vals2', vals)
        vals=vals*valsmask
        #print('vals3', vals)
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data

def evaluate_cbfdotlatentbiased_func(cbfdot,
                             env,
                             file=None,
                             plot=True,
                             show=False,
                             skip=1,
                             action=(0,0)):#(1,1)):#2):#
    data = np.zeros((spb.WINDOW_HEIGHT, spb.WINDOW_WIDTH))
    #if walls is None:
    xmove = 0  # -25#30#
    ymove = 0  #-45#-40#-35#-33#-30#-25#
    lux = 50#75#105#
    luy = 55#40#
    width = 25#20#
    height = 40#50#
    walls = [((lux + xmove, luy + ymove), (lux + width + xmove, luy + height + ymove))] #[((75, 55), (100, 95))] #the position and dimension of the wall
    #self.walls = [self._complex_obstacle(wall) for wall in walls]  # 140, the bound of the wall
    # it is a list of functions that depend on states
    selfwall_coords = np.array(walls)
    for y in tqdm(range(0, spb.WINDOW_HEIGHT, skip)):
        row_states = []
        #row_statesu = []
        for x in range(0, spb.WINDOW_WIDTH, skip):
            old_state = np.array((x,y))#env._state_to_image((x, y)) / 255
            old_stateimage = env._state_to_image((x, y)) / 255#, this is for using the global coordinate
            #old_stateimage = env._state_to_image_relative((x, y)) / 255#, this is for using the ego coordinate
            row_states.append(old_stateimage)

        #vals = cbfdot.cbfdots(np.array(row_states)).squeeze()#it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots(np.array(row_states),already_embedded = True).squeeze()  # it is like calling forward of const_estimator!
        vals = cbfdot.cbfdots(np.array(row_states)).squeeze() #latent and latent plan a are the same in this case # it is like calling forward of const_estimator!
        #vals = cbfdot.cbfdots_planb(np.array(row_states)).squeeze()  #
        #print('vals',vals)
        #valsu=np.array(row_statesu).squeeze()#print('valsu', valsu)
        #vals=(vals)**3#only applies to plan b!!! not for plan a!!!#-valsu#print('valsnew', vals)
        #print('vals1',vals)
        #valsmask=np.where(vals<=0,-1,1)
        #vals=(np.abs(vals))**(coeff)#np.power(vals,1/3,dtype=complex)
        #print('vals2', vals)
        #vals=vals*valsmask
        #print('vals3', vals)
        if skip == 1:
            data[y] = vals.squeeze()
        elif skip == 2:
            data[y, ::2], data[y, 1::2] = vals, vals,
            data[y + 1, ::2], data[y + 1, 1::2] = vals, vals
        else:
            raise NotImplementedError("[name redacted :)] has not implemented logic for skipping %d yet" % skip)

    if plot:
        env.draw(heatmap=data, file=file, show=show, board=False)

    return data