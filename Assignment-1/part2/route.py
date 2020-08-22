#!/usr/local/bin/python3
#
# Code by: [Vidit Mohaniya - Pauravi Wagh - Vibhas Vats vimohan - pwagh - vkvats]
#


import heapq
import math
import sys


# To Print object's all values while debugging
def auto_str(cls):
    def __str__(self):
        return '%s(%s)' % (
            type(self).__name__,
            ', '.join('%s=%s' % item for item in vars(self).items())
        )
    cls.__str__ = __str__
    cls.__repr__ = __str__
    return cls

# To Store All Cities
# @auto_str
class City:
    def __init__(self, name, latitude = 0, longitude = 0):
        self.name = name
        self.latitude = float(latitude)
        self.longitude = float(longitude)
        self.adjacent = {}

    def connect_highway(self, neighbor, edge):
        self.adjacent[neighbor.name] = edge

    def get_connecting_highways(self):
        return self.adjacent.keys()

    def __str__(self):
        return str(self.name)
               # + ' adjacent: ' + str([x.id for x in self.adjacent])

    def __repr__(self):
        return str(self.name)

# To Store All Edges
@auto_str
class Highway:
    def __init__(self, first_city, second_city, length, speed_limit, name):
        self.name = name
        self.first_city = first_city
        self.second_city = second_city
        self.length = int(length)
        self.speed_limit = float(speed_limit)
        self.mpg = (400/150) * self.speed_limit * (1 - self.speed_limit/150) ** 4


# To Store Road-Segments Network
@auto_str
class Graph:
    def __init__(self):
        self.vertices = {}

    def add_city(self, city):
        self.vertices[city.name] = city
        return city

    def add_highway(self, frm, to, edge, edge_reverse):
        self.vertices[frm.name].connect_highway(self.vertices[to.name], edge)
        self.vertices[to.name].connect_highway(self.vertices[frm.name], edge_reverse)

    def get_city(self, n):
        if n in self.vertices:
            return self.vertices[n]
        else:
            return None

    def __iter__(self):
        return iter(self.vertices.values())


# To store generated solution at every state
# @auto_str
class Solution:

    def __init__(self, total_segments = 0, total_miles = 0, total_hours = 0, total_gas_gallons = 0, cities_travelled = []):
        self.total_segments = total_segments
        self.total_miles = total_miles
        self.total_hours = total_hours
        self.total_gas_gallons = total_gas_gallons
        self.cities_travelled = cities_travelled

    def __str__(self):
        return str(str(self.total_segments) + ' ' + str(self.total_miles) + ' ' + str(round(self.total_hours, 4)) + ' ' + str(round(self.total_gas_gallons, 4)) \
                   + ' ' + ' '.join(self.cities_travelled))


# To estimate cost at the given state
def cost_estimation(cost_function, counter, sol):
    if cost_function == 'segments':
        return counter + 1
    elif cost_function == 'distance':
        return sol.total_miles
    elif cost_function == 'time':
        return sol.total_hours
    elif cost_function == 'mpg':
        return sol.total_gas_gallons


def load_road_segments(filename, gpsfilename):
    roads=Graph()
    gps = {}
    with open(gpsfilename, "r") as file:
        for line in file:
            l = line.split()
            gps.update({l[0]: [l[1], l[2]]})

    with open(filename, "r") as file:
        for line in file:
            l = line.split()
            gps_first_city = gps.get(l[0])
            gps_second_city = gps.get(l[1])
            if gps_first_city:
                first_city = City(l[0], gps_first_city[0], gps_first_city[1])
            else:
                first_city = City(l[0])
            if gps_second_city:
                second_city = City(l[1], gps_second_city[0], gps_second_city[1])
            else:
                second_city = City(l[1])

            road = Highway(first_city, second_city, l[2], l[3], l[4])
            road_reverse = Highway(second_city, first_city, l[2], l[3], l[4])
            if not roads.get_city(first_city.name):
                roads.add_city(first_city)
            if not roads.get_city(second_city.name):
                roads.add_city(second_city)
            roads.add_highway(first_city, second_city, road, road_reverse)
    return roads


def updated_sol(sol, edge):
    new_sol = Solution()
    new_sol.total_segments = sol.total_segments + 1
    new_sol.cities_travelled = new_sol.cities_travelled + sol.cities_travelled
    new_sol.cities_travelled.append(edge.second_city.name)
    new_sol.total_gas_gallons = sol.total_gas_gallons + edge.length / edge.mpg
    new_sol.total_miles = sol.total_miles + edge.length
    new_sol.total_hours = sol.total_hours + (edge.length / edge.speed_limit)
    return new_sol


def get_connected_cities(roads, city_sol):
    cities = []
    city = city_sol[0]
    c = roads.get_city(city)
    for neighbour in c.adjacent:
        sol = city_sol[1]
        cities.append((neighbour, updated_sol(sol, c.adjacent[neighbour])))

    return cities


def search(roads, start_city, end_city, cost_function):
    fringe = []
    counter = 1
    break_prioiry = 1
    heapq.heappush(fringe, (counter, break_prioiry, (start_city, Solution(cities_travelled = [start_city]))))
    visited_cities = {}
    visited_priority = {}
    destination = roads.get_city(end_city)
    while fringe:
        cities = get_connected_cities(roads, heapq.heappop(fringe)[-1])
        for c, sol in cities:
            if c == end_city:
                # sol.cities_travelled.append(end_city)
                return sol
            break_prioiry += 1
            counter = cost_estimation(cost_function, counter, sol)
            if visited_cities.get(c) and visited_priority.get(c) and visited_priority[c] > counter:
                visited_cities.pop(c)
                visited_priority.update({c: counter})
                heapq.heappush(fringe, (counter, break_prioiry, (c, sol)))
            elif not visited_cities.get(c):
                visited_priority.update({c: counter})
                heapq.heappush(fringe, (counter, break_prioiry, (c, sol)))
                visited_cities.update({c: True})


def human_readable_format(solution):
    print('Total Segments: ', solution.total_segments)
    print('Total Distance: ', solution.total_miles)
    print('Total Time Taken: ', round(solution.total_hours, 4))
    print('Total Gallons Needed: ', round(solution.total_gas_gallons, 4))
    print('Cities Travelled in Order: ', solution.cities_travelled)


if __name__ == "__main__":
    roads = load_road_segments('road-segments.txt', 'city-gps.txt')
    if (len(sys.argv) != 4):
        raise (Exception("Error: expected 3 arguments"))

    solution = search(roads, sys.argv[1], sys.argv[2], sys.argv[3])
    human_readable_format(solution)
    print(solution if solution else 'Inf')
