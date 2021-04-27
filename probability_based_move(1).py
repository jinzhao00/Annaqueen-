 
#----- IFN680 Assignment 1 -----------------------------------------------#
#  The Wumpus World: a probability based agent
#
#  Implementation of two functions
#   1. PitWumpus_probability_distribution()
#   2. next_room_prob()
#
#    Student no: n9754911
#    Student name: Jicheng Peng
#
#-------------------------------------------------------------------------#
from random import *
from AIMA.logic import *
from AIMA.utils import *
from AIMA.probability import *
from tkinter import messagebox

#--------------------------------------------------------------------------------------------------------------
#
#  The following two functions are to be developed by you. They are functions in class Robot. If you need,
#  you can add more functions in this file. In this case, you need to link these functions at the beginning
#  of class Robot in the main program file the_wumpus_world.py.
#
#--------------------------------------------------------------------------------------------------------------
#   Function 1. PitWumpus_probability_distribution(self, width, height)
#
# For this assignment, we treat a pit and the wumpus equally. Each room has two states: 'empty' or 'containing a pit or the wumpus'.
# A Boolean variable to represent each room: 'True' means the room contains a pit/wumpus, 'False' means the room is empty.
#
# For a cave with n columns and m rows, there are totally n*m rooms, i.e., we have n*m Boolean variables to represent the rooms.
# A configuration of pits/wumpus in the cave is an event of these variables.
#
# The function PitWumpus_probability_distribution() below is to construct the joint probability distribution of all possible
# pits/wumpus configurations in a given cave, two parameters
#
# width : the number of columns in the cave
# height: the number of rows in the cave
#
# In this function, you need to create an object of JointProbDist to store the joint probability distribution and  
# return the object. The object will be used by your function next_room_prob() to calculate the required probabilities.
#
# This function will be called in the constructor of class Robot in the main program the_wumpus_world.py to construct the
# joint probability distribution object. Your function next_room_prob() will need to use the joint probability distribution
# to calculate the required conditional probabilities.
#
def PitWumpus_probability_distribution(self, width, height): 
    # Create a list of variable names to represent the rooms. 
    # A string '(i,j)' is used as a variable name to represent a room at (i, j)
    self.PW_variables = [] 
    for column in range(1, width + 1):
        for row in range(1, height + 1):
            self.PW_variables  = self.PW_variables  + ['(%d,%d)'%(column,row)]
            
    #--------- Add your code here -------------------------------------------------------------------
     
    Var_val = {each:[True,False] for each in self.PW_variables} 
    Pr = JointProbDist(self.PW_variables, Var_val) 
    print(Pr.show_approx())
    return Pr
    

       
            
        
#---------------------------------------------------------------------------------------------------
#   Function 2. next_room_prob(self, x, y)
#
#  The parameters, (x, y), are the robot's current position in the cave environment.
#  x: column
#  y: row
#
#  This function returns a room location (column,row) for the robot to go.
#  There are three cases:
#
#    1. Firstly, you can call the function next_room() of the logic-based agent to find a
#       safe room. If there is a safe room, return the location (column,row) of the safe room.
#    2. If there is no safe room, this function needs to choose a room whose probability of containing
#       a pit/wumpus is lower than the pre-specified probability threshold, then return the location of
#       that room.
#    3. If the probabilities of all the surrounding rooms are not lower than the pre-specified probability
#       threshold, return (0,0).
#
def next_room_prob(self, x, y):
    #messagebox.showinfo("Not yet complete", "You need to complete the function next_room_prob.")
    #pass
    #--------- Add your code here -------------------------------------------------------------------
    T,F=True,False
    P_false= 1-0.2 
    P_true=0.2 
    
    Pr = self.jdP_PWs
    print(Pr.show_approx())
   
    
    if self.next_room(x,y)==(0,0): 
        

        #Pr = self.PitWumpus_probability_distribution.show_approx()

        visited_rooms = []
        for each in self.visited_rooms:
            visited_rooms.append(str(each).replace(' ', ''))
        print ('vsroom'+str(visited_rooms))
        sur_rooms = []
        for each in self.cave.getsurrounding(x,y):
            sur_rooms.append(str(each).replace(' ', '') )
        print ('vsroom'+str(sur_rooms))


        l=visited_rooms
        m=self.PW_variables
        s=sur_rooms
        frontier = [each for each in sur_rooms]
        print('frontier rooms'+ str(frontier))
        if len(frontier) == 0:
            return (0,0) 
        
        
        frontier_result={}

        
        
        
        for each in frontier:
            each_tuple = each
            
            frontier_copy = [each for each in frontier]
            print(str(frontier_copy))
            current_frontier = each_tuple
            current_frontier_true = all_events_jpd(frontier_copy, self.jdP_PWs, {each_tuple: True})
            p_true = 0
            for each_true in current_frontier_true:
                for (var, val) in each_true.items(): 
                    store = {str(frontier):True}
                    p_event = val
                    store[var] = val
                    p_event*self.consistent(self.observation_breeze_stench(visited_rooms),store)
                    p_true += p_event
                    
                    
            p_true*=0.2
            
            
            current_query_false = all_events_jpd(frontier_copy, Pr, {each_tuple: False})
            p_false = 0
            for each_false in current_query_false:
                for (var, val) in each_false.items():
                    store = {str(frontier):False}
                    p_event = val
                    store[var] = val
                   p_event*self.consistent(self.observation_breeze_stench(visited_rooms),store)
                    p_false += p_event
                    
                    
            p_false*=0.8
            p_true_Normalized = p_true/(p_true+p_false)
            p_false_Normalized = p_false/(p_true+p_false)
            print('each_tuple' + str(each_tuple))
            frontier_result[each_tuple]=p_true_Normalized
            
            
        key_result = (0,0)
        value_result = 1
        
            
        for (var,val) in frontier_result.items():
            
            
            if (val < value_result) & (val < self.max_pit_probability):
                print (var)
                key_result = set(var)
             

       
        return key_result
        #if next_room_prob > self.max_pit_probability:
        #    return (0, 0)
        #else :
        #    key = [k for k, v in frontier_result.items() if v==next_room_prob]
         #   next_room = key[0]
        #    return(next_room)
            
            
            
        
        
        

    else:
        return self.next_room(x,y) 


#---------------------------------------------------------------------------------------------------
 
####################################################################################################
