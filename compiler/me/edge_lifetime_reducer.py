'''Edge Lifetime Reducer (EuLeR)'''
import json
import networkx as nx
import numpy as np
import sys
import argparse
import re
import logging, sys

TOTAL_EVENTS = 256
RESERVED_EVENTS = 13
MAX_EVENTS = (TOTAL_EVENTS - RESERVED_EVENTS)
LEVEL_PRINTING_STEP = 1000
MAX_ELR_CNT = 5

class EdgeLifetimeReducer():
    class Metrics():
        def __init__ (self, name):
            self.serial_cnt = 0
            self.avg_live_edges = 0
            self.total_depths = 0
            self.max_live_edges = 0
            self.total_live_edges = 0
            self.name = name
            self.level2num_liveedges_nonmem2mem = []
            self.level2num_liveedges_mem2nonmem = []
            self.level2num_liveedges_mem2mem = []
            self.level2num_liveedges_nonmem2nonmem = []

        def __str__ (self):
            res = "==========================================================\n"
            res += "= " + self.name+"\n"
            res += ("INFO:Total number of Serialization = %d\n"%self.serial_cnt)
            res += ("INFO:Average number of live edges per depth = %f\n"%\
                (self.total_live_edges / self.total_depths))
            res += ("INFO:Maximum number of live edges per depth = %d\n"%\
                self.max_live_edges)
            res += ("INFO:Total depths = %d\n"%self.total_depths)
            return res
        def compare(self, base_metrics):
            delimit =\
                "=========================================================="
            print("%s"%delimit)
            print("INFO:Change of average live edges = %f %%"%\
                ((base_metrics.avg_live_edges - self.avg_live_edges)*100/
                    base_metrics.avg_live_edges))
            print("INFO:Change of total depth = %f %%"%\
                ((-base_metrics.total_depths + self.total_depths)*100/\
                    base_metrics.total_depths))
            print("INFO:Change of max live edges = %f %%"%\
                ((base_metrics.max_live_edges - self.max_live_edges)*100/\
                    base_metrics.max_live_edges))

    def __init__(
            self
            , wavegraph_file
            , max_edges_level
            , num_edges_consolidate
            , synthetic_graph
            , run_until_end
        ):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.before_metrics = self.Metrics("Before EuLeR")
        self.after_metrics = self.Metrics("After EuLeR")
        self.synthetic_graph = synthetic_graph
        self.max_edges_level = max_edges_level
        self.num_edges_consolidate = num_edges_consolidate
        self.run_until_end = run_until_end
        with open(wavegraph_file) as f:
            self.wavegraph_json = json.load(f)
        self.vtx2wop_name = []
        self.vtx2wop_type = []
        self.wop_name2vtx = dict()
        self.vtx2wop = []
        logging.debug("Starting transform_wavegraph_to_nxgraph")
        self.nx_wavegraph = self.transform_wavegraph_to_nxgraph()
        logging.debug("Finished transform_wavegraph_to_nxgraph")
        self.initialize(first = True)
        self.profile_edge_types()
#        self.compute_levels()
#        self.level2vtx_list = []
#        self.level2edge_list = []
#        for i in range(self.max_level):
#            self.level2vtx_list.append([])
#            self.level2edge_list.append([])
#        for n in self.nx_wavegraph.nodes():
#            level = int(self.vtx2level[n])
#            self.level2vtx_list[level].append(n)
#        self.edge_cnt_level = []
#        self.compute_lifetime(self.nx_wavegraph)

    def initialize (self, first):
#        print ("INFO:Initializing containers and computing lifetimes...")
        if (first):
            self.level2vtx_list = []
            self.level2edge_list = []
            self.edge_cnt_level = []
            self.edges = []
            for e in self.nx_wavegraph.edges():
                self.edges.append(e)
        logging.debug("Computing levels...")
        self.compute_levels(self.nx_wavegraph, first)
        logging.debug("Finished compute_levels...")
        if (first):
            for i in range(self.max_level):
                self.level2vtx_list.append([])
                self.level2edge_list.append([])
                self.before_metrics.level2num_liveedges_mem2mem.append(0)
                self.before_metrics.level2num_liveedges_mem2nonmem.append(0)
                self.before_metrics.level2num_liveedges_nonmem2mem.append(0)
                self.before_metrics.level2num_liveedges_nonmem2nonmem.append(0)
                self.after_metrics.level2num_liveedges_mem2mem.append(0)
                self.after_metrics.level2num_liveedges_mem2nonmem.append(0)
                self.after_metrics.level2num_liveedges_nonmem2mem.append(0)
                self.after_metrics.level2num_liveedges_nonmem2nonmem.append(0)
            for n in self.nx_wavegraph.nodes():
                level = int(self.vtx2level[n])
                self.level2vtx_list[level].append(n)
        logging.debug("Computing lifetimes...")
        self.compute_lifetime(self.nx_wavegraph, first)
        logging.debug("Done initialization...")

    def count_waveops_zero_incoming_edge (self):
        assert(self.wavegraph_json["waveops"] != None)
        cnt = 0
        for wop in self.wavegraph_json["waveops"]:
            if (len(wop["previous_waveops"]) == 0):
                cnt += 1
        return cnt

    def transform_wavegraph_to_nxgraph(self):
#        print ("INFO:Trasforming wavegraph to nx_wavegraph...")
        wavegraph_nx = nx.DiGraph()
        if (self.synthetic_graph == False):
            vtx = 0
            for wop in self.wavegraph_json["waveops"]:
                self.wop_name2vtx[wop["waveop_name"]] = vtx
                self.vtx2wop_name.append(wop["waveop_name"])
                self.vtx2wop_type.append(wop["waveop_type"])
                self.vtx2wop.append(wop)
                for p in wop["previous_waveops"]:
                    p_vtx = self.wop_name2vtx[p]
                    assert(p_vtx != None)
                    wavegraph_nx.add_edge(p_vtx, vtx, lifetime=0)
                vtx += 1
        else:
            self.create_synthetic1(wavegraph_nx)
        return wavegraph_nx
    
    def create_synthetic1 (self, nx_wavegraph):
        nx_wavegraph.add_edge(0, 5, lifetime=0)
        nx_wavegraph.add_edge(1, 3, lifetime=0)
        nx_wavegraph.add_edge(3, 4, lifetime=0)
        nx_wavegraph.add_edge(4, 5, lifetime=0)
        nx_wavegraph.add_edge(2, 5, lifetime=0)

    def profile_edge_types (self):
        def is_mem_type(v):
            return (self.vtx2wop_type[v] == "SBAtomSave" or\
             self.vtx2wop_type[v] == "SBAtomLoad")
        for e in self.nx_wavegraph.edges():
            source_level = self.vtx2level[e[0]]
            target_level = self.vtx2level[e[1]]
            source_type = is_mem_type(e[0])
            target_type = is_mem_type(e[1])
            for l in range(source_level, target_level):
                if (source_type and target_type):
                    self.before_metrics.level2num_liveedges_mem2mem[l] += 1
                elif (source_type and not target_type):
                    self.before_metrics.level2num_liveedges_mem2nonmem[l] += 1
                elif (not source_type and target_type):
                    self.before_metrics.level2num_liveedges_nonmem2mem[l] += 1
                else:
                    self.before_metrics.level2num_liveedges_nonmem2nonmem[l]+=1

    def compute_max_fanout(self):
        max_fanout = 0
        max_fanout_wop = ""
        for v in self.nx_wavegraph.nodes():
            if (self.nx_wavegraph.out_degree(v) > max_fanout):
                max_fanout = self.nx_wavegraph.out_degree(v)
                max_fanout_wop = self.vtx2wop_name[v]
        return (max_fanout, max_fanout_wop)

    def compute_levels (self, nx_wavegraph, first):
        if (first):
            self.vtx2level = []
            for i in range(len(nx_wavegraph.nodes())):
                self.vtx2level.append(0)
        max_level = self.asap(self.nx_wavegraph)
        self.max_level = int(max_level) + 1
        return int(max_level) + 1

    def asap (self, nx_graph):
        logging.debug("ASAP scheduling...")
        max_level = 0
        vs = nx.topological_sort(nx_graph)
        logging.debug("TOPO sort is done...")
        for v in vs:
            for p in nx_graph.predecessors(v):
                self.vtx2level[v] =\
                    int(max(self.vtx2level[v], self.vtx2level[p] + 1))
            max_level = max(max_level, self.vtx2level[v])
        logging.debug("Finished ASAP scheduling...")
#        print ("max_level = %d"%max_level)
        return max_level

    def compute_lifetime (self, nx_wavegraph, first):
        logging.debug("Starting compute_lifetime...")
        if (first):
            for l in range(self.max_level):
                self.edge_cnt_level.append(0)
        else:
            append_cnt = self.max_level - len(self.edge_cnt_level)
#            print("append_cnt = %d"%append_cnt)
            for i in range(append_cnt):
                self.edge_cnt_level.append(0)
#            for l in range(self.max_level):
#                self.edge_cnt_level[l] = 0
        if (first):
            logging.info("Num Edges = %d"%len(nx_wavegraph.edges()))
#            for (u,v) in nx_wavegraph.edges():
            for eid in range(len(self.edges)):
                u = self.edges[eid][0]
                v = self.edges[eid][1]
                start_level = self.vtx2level[u]
                end_level = self.vtx2level[v]
                nx_wavegraph[u][v]['lifetime'] = end_level - start_level
#                if (self.vtx2wop_type[u] != self.vtx2wop_type[v]):
                for i in range(start_level, end_level):
                    self.edge_cnt_level[i] += 1
                    self.level2edge_list[i].append(eid)
        logging.debug("Finishied compute_lifetime...")
#        for l in range(self.max_level):
#            print ("level %d : # Crossing Edges = %d"%(l, edge_cnt_level[l]))

    def count_ld_level (self, nx_wavegraph):
        for l in range(self.max_level):
            ld_cnt = 0
            for op in self.level2vtx_list[l]:
                if (self.vtx2wop_type[op] == "SBAtomLoad"):
                    ld_cnt += 1
            print ("level %d : # LDs = %d"%(l, ld_cnt))

    def lifetime_edges (self, nx_wavegraph):
        lt = []
        for (u,v) in nx_wavegraph.edges():
#            print ((self.vtx2level[v] - self.vtx2level[u]))
            lt_ = (self.vtx2level[v] - self.vtx2level[u])
            if (lt_ > 1000):
                print ("lifetime = %d"%lt_, end = " ")
                print ("%s -> %s"%(
                    self.vtx2wop_name[u], self.vtx2wop_name[v]))
            lt.append(lt_)
        print (np.histogram(lt))
#        import matplotlib.pyplot as plt
#        plt.hist(lt, bins='auto')  # arguments are passed to np.histogram
#        plt.title("Histogram with 'auto' bins")
#        plt.show()
    def serialize_all_loads (self, nx_wavegraph):
        prev_ld = None
        for v in nx.topological_sort(nx_wavegraph):
            if (self.vtx2wop_type == "SBAtomLoad"):
                if (prev_ld != None):
                    nx_wavegraph.add_edge(prev_ld, v)
                prev_ld = v

    def get_edge_tuples (self, eids):
        edges = []
        for eid in eids:
            edges.append(self.edges[eid])
        return edges
    def gather_longest_edges(self, level):
        def get_remaining_lifetime(edge):
            start_level = edge[0]
            end_level = edge[1]
            if (level < start_level): lt = end_level - start_level
            elif (level > end_level): lt = 0
            else: lt = end_level - level
            return lt
        edges = self.get_edge_tuples(self.level2edge_list[level])
        edges.sort(key=get_remaining_lifetime)
        cnt = 0
        result = []
        latest_source_level = 0
        earliest_target_level = len(self.nx_wavegraph.nodes())
        for e in edges:
            if (self.nx_wavegraph.has_edge(e[0], e[1]) and\
                self.vtx2wop_type[e[1]] != "SBAtomSave"):
                if (self.vtx2wop_type[e[0]] != "MatMul" or\
                    self.vtx2wop_type[e[1]] != "MatMul" or\
                    (self.vtx2wop[e[1]])["weights_sb_address"] != -1):
                    if (self.vtx2level[e[0]] < earliest_target_level and\
                        self.vtx2level[e[1]] > latest_source_level):
                        result.append(e)
                        cnt += 1
                        latest_source_level =\
                            int(max(latest_source_level, self.vtx2level[e[0]]))
                        earliest_target_level =\
                            int(min(earliest_target_level
                                    , self.vtx2level[e[1]]))
                        if (cnt == self.num_edges_consolidate):
                            break
#        return edges[0:self.num_edges_consolidate - 1]
        return result

    def serialize_vertices (self, edges, serialize_sources):
#        print("INFO:Serializing vertices")
        vertices = []
        def get_level(v):
            return self.vtx2level[v]
        for (u, v) in edges:
            if (serialize_sources == True):
                if (not u in vertices):
                    vertices.append(u)
            else:
                if (not v in vertices):
                    vertices.append(v)
        vertices.sort(key=get_level)
#        print("serialize_sources = ", serialize_sources)
#        print ([(v, self.vtx2level[v]) for v in vertices])
        pred = vertices[0]
#        level = self.vtx2level[pred]
        for v in vertices[1:]:
            if (v != pred):
                if (not self.nx_wavegraph.has_edge(pred, v) and\
                    not self.nx_wavegraph.has_edge(v, pred)):
                    self.nx_wavegraph.add_edge(pred, v)
    #                print ("added edge between ", (pred, v))
    #                self.level2vtx_list[self.vtx2level[v]].remove(v)
    #                self.vtx2level[v] = level + 1
    #                self.level2vtx_list[level + 1].append(v)
                    self.edge_cnt_level[self.vtx2level[pred]] += 1
#                    self.level2edge_list[self.vtx2level[pred]].append((pred, v))
    #                print("serialze_vertices::pred = %d v = %d"%(pred, v))
    #                print("serialze_vertices::edge_cnt_level[%d] = %d"%(
    #                    (level), self.edge_cnt_level[level]))
    #                level += 1
            pred = v
#        print("INFO:Finished serializing vertices")
        return vertices[-1] if (serialize_sources) else vertices[0]

    # Adjust edge_cnt_level at levels 'edges' are associated with.
    # Since we will remove/add 'edges', edge_cnt_level should be adjusted
    # accordingly.
    def adjust_edge_cnt_level (self, edges, increment):
        for (u, v) in edges:
            start_level = self.vtx2level[u]
            end_level = self.vtx2level[v]
#            print ("(u,v) = ", (u, v))
#            print ("start level = %d"%start_level, end = " ")
#            print ("end level = %d"%end_level)
            for l in range(start_level, end_level):
#                self.level2edge_list[l].append((u,v))
                if (increment):
                    self.edge_cnt_level[l] += 1
                else:
                    self.edge_cnt_level[l] -= 1
#                print ("l = %d"%l)
#                print ("edge_cnt_level[%d] = %d"%(l, self.edge_cnt_level[l]))

    def add_one_long_edge (self, s, t):
#        print ("(s, t) = ", (s, t))
        assert(not self.nx_wavegraph.has_edge(t, s))
        lt = self.vtx2level[t] - self.vtx2level[s]
        self.nx_wavegraph.add_edge(s, t, lifetime=lt)
        e = (s, t) # Added edge
        self.adjust_edge_cnt_level([e], True) # Increment

    def limit_max_edges_per_level (self):
        first = True
        elr_cnt = 0
        self.after_metrics.serial_cnt = 0
        serial_cnt = 0
        changed = False
        while((first or serial_cnt > 0) and\
            ((self.run_until_end == False and elr_cnt < MAX_ELR_CNT) or\
            self.run_until_end == True)):
            logging.info("elr_cnt = %d"%elr_cnt)
            serial_cnt = 0
            for l in range(self.max_level):
                if (l % LEVEL_PRINTING_STEP == 0):
                    logging.info("Starting level %d"%l)
                if (self.edge_cnt_level[l] > self.max_edges_level):
                    serial_cnt += 1
                    changed = True
                    edges = self.gather_longest_edges(l)
                    if (len(edges)):
                        last_source = self.serialize_vertices(edges, True)
                        first_target = self.serialize_vertices(edges, False)
                        self.remove_edges(edges)
                        self.add_one_long_edge(last_source, first_target)
                        self.initialize(first = False)
            self.after_metrics.serial_cnt += serial_cnt
            first = False
            elr_cnt += 1
            self.initialize(first = True)
        return changed

    def remove_edges(self, edges):
        for e in edges:
            start_level = self.vtx2level[e[0]]
            end_level = self.vtx2level[e[1]]
#            print ("e = ", e)
            for l in range(start_level, end_level):
#                self.level2edge_list[l].remove(e)
                self.edge_cnt_level[l] -= 1
        self.nx_wavegraph.remove_edges_from(edges)

    def print_edges_per_level (
            self, filename, before = True, complete_init = False
        ):
        self.initialize(first = complete_init)
#        print ("depth, Num live edges")
        total_live_edges = 0
        max_le = 0
        dist = str()
        f = open(filename,"w")
        if (before):
            metrics = self.before_metrics
        else:
            metrics = self.after_metrics
        for l in range(self.max_level):
#            print("%d, %d"%(l, len(self.level2edge_list[l])))
            dist += ("%d, %d, "%(l, len(self.level2edge_list[l])))
            dist += ("%d, "%(metrics.level2num_liveedges_mem2mem[l]))
            dist += ("%d, "%(metrics.level2num_liveedges_mem2nonmem[l]))
            dist += ("%d, "%(metrics.level2num_liveedges_nonmem2mem[l]))
            dist += ("%d\n"%(metrics.level2num_liveedges_nonmem2nonmem[l]))
            max_le = int(max(max_le, len(self.level2edge_list[l])))
            total_live_edges += len(self.level2edge_list[l])
        f.write(dist)
        f.close()
        if (before):
            self.before_metrics.total_live_edges = total_live_edges
            self.before_metrics.avg_live_edges= (total_live_edges/self.max_level)
            self.before_metrics.total_depths = self.max_level
            self.before_metrics.max_live_edges = max_le
        else:
            self.after_metrics.total_live_edges = total_live_edges
            self.after_metrics.avg_live_edges= (total_live_edges/self.max_level)
            self.after_metrics.total_depths = self.max_level
            self.after_metrics.max_live_edges = max_le

    def convert_nx_to_wavegraph (self, changed):
        new_stream = []
        if (changed):
          for v in nx.topological_sort(self.nx_wavegraph):
              wop = self.vtx2wop[v]
              prevs = []
              for in_e in self.nx_wavegraph.in_edges(v):
                  prev_wop = self.vtx2wop_name[in_e[0]]
                  prevs.append(prev_wop)
              wop["previous_waveops"] = prevs
              new_stream.append(wop)
          self.wavegraph_json["waveops"] = new_stream
        print("Saving Wave-Graph %s"%(args.wavegraph+"-elr"))
        with (open(args.wavegraph+"-elr", 'w')) as f:
            s = json.dumps(self.wavegraph_json, indent=2, sort_keys=True)
            s = re.sub(r'\s+(\d+,)\n\s+(\d+)', r'\1\2', s, flags=re.S)
            s = re.sub(r',\s*(\d+)\n\s+\]', r',\1]', s, flags=re.S)
            f.write(s)
        


if (__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavegraph", default="wavegraph.json"\
        , help="Wave-graph Json file to read; defaults to wavegraph.json")
    parser.add_argument("--max_events", type=int, default=MAX_EVENTS\
        , help="Maxium number of events")
    parser.add_argument("--num_edges_consolidate", type=int, default=10\
        , help="Number of edges to consolidate into one long edge")
    parser.add_argument("--synthetic", type=bool, default=False\
        , help = "Use a synthetic graph for test purpose")
#    parser.add_argument("--live_edge_dist_only", type=bool, default=False\
#        , help =\
#            "Prints only live edge distribution without edge lifetime reduction"
#    )
    parser.add_argument("--before_dist", default="org.csv"\
     , help="CSV file name for storing live edge distribution before EuLeR")
    parser.add_argument("--after_dist", default="opt.csv"\
     , help="CSV file name for storing live edge distribution after EuLeR")
    parser.add_argument("--run_until_end", type=bool, default=False\
        ,help = "Ignore limit of EuLeR run and run until max_events is reached")
    parser.add_argument("--profile_only", action='store_true', default = False) 
    args = parser.parse_args()
    l = EdgeLifetimeReducer(
            args.wavegraph
            , args.max_events
            , args.num_edges_consolidate
            , args.synthetic
            , args.run_until_end
        )
#    print (l.count_waveops_zero_incoming_edge())
#    print (l.compute_max_fanout())
#    l.num_edges_crossing_levels(l.nx_wavegraph)
#    l.count_ld_level(l.nx_wavegraph)
#    l.lifetime_edges(l.nx_wavegraph)
    l.print_edges_per_level(args.before_dist)
    if (not args.profile_only):
        changed = l.limit_max_edges_per_level()
        l.convert_nx_to_wavegraph(changed)
        l.print_edges_per_level(args.after_dist, before=False, complete_init = True)
        print("%s"%l.before_metrics)
        print("%s"%l.after_metrics)
        l.after_metrics.compare(l.before_metrics)
