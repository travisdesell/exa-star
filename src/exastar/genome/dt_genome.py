from __future__ import annotations
from itertools import chain, product
import math
from typing import cast, Dict, List, Set

from exastar.genome.visitor.reachability_visitor import ReachabilityVisitor
from exastar.weights import WeightGenerator
from genome import MSEValue
from exastar.genome.component.dt_node import DTNode
from exastar.genome.component.dt_input_node import DTInputNode
from exastar.genome.component.dt_output_node import DTOutputNode
from exastar.genome.component.dt_set_edge import DTBaseEdge
from exastar.genome.exastar_genome import EXAStarGenome
from exastar.time_series import TimeSeries
from genome import FitnessValue
from util.functional import is_not_any_type

import bisect
from loguru import logger
import numpy as np
import torch
import random

from util.typing import overrides


class DTGenome(EXAStarGenome[DTBaseEdge]):

    @staticmethod
    def make_trivial(
            generation_number: int,
            input_series_names: List[str],
            output_series_names: List[str],
            weight_generator: WeightGenerator,
            rng: np.random.Generator,
            guide=None,
    ) -> DTGenome:
        input_nodes = {
            "Start": DTInputNode("Start", 0.0)
        }
        options = input_series_names
        output_nodes = {}
        guide = guide
        for output_name in output_series_names:
            output_nodes[output_name] = DTOutputNode(output_name, 1.0)
            output_nodes["Hold"] = DTOutputNode("Hold", 1.0)

        edges: List[DTBaseEdge] = [
            DTBaseEdge(input_nodes["Start"], output_nodes[output_series_names[0]], True)
        ]

        nodes: List[DTNode] = list(chain(input_nodes.values(), output_nodes.values()))
        logger.info(f"output series: {output_series_names}")

        g = DTGenome(
            generation_number,
            list(input_nodes.values()),
            list(output_nodes.values()),
            nodes,
            edges,
            MSEValue(math.inf),
            options,
            guide
        )

        weight_generator(g, rng)
        return g

    def __init__(
            self,
            generation_number: int,
            input_nodes: List[DTInputNode],
            output_nodes: List[DTOutputNode],
            nodes: List[DTNode],
            edges: List[DTBaseEdge],
            fitness: FitnessValue,
            options: List[str],
            guide,
    ) -> None:
        """
        Initialize base class genome fields and methods.
        Args:
            generation_number: is a unique number for the genome,
              generated in the order that they were created. higher
              genome numbers are from genomes generated later
              in the search.
        """
        EXAStarGenome.__init__(
            self,
            generation_number,
            input_nodes,
            output_nodes,
            nodes,
            edges,
            fitness,
        )
        self.options = options
        self.guide = guide

    def sanity_check(self):
        """
        TODO: Update for DT
        """
        # Ensure all edges referenced by nodes are in self.edges
        # and vica versa
        edges_from_nodes: Set[DTBaseEdge] = set()
        for node in self.nodes:
            edges_from_nodes.update(node.input_edges)
            edges_from_nodes.update(node.output_edges)

            # Check for orphaned nodes
            if not isinstance(node, DTInputNode):
                assert node.input_edges

            # Input nodes can be orphaned; output nodes need not have output edges. to not be orphaned.
            if not isinstance(node, DTInputNode) and not isinstance(node, DTOutputNode):
                assert node.output_edges

            assert node.weights_initialized(), f"node = {node}"

        aa = set(self.edges)
        bb = set(self.inon_to_edge.values())
        assert aa == edges_from_nodes == bb, f"{aa}\n{bb}\n{edges_from_nodes}"

        assert set(self.nodes) == set(self.inon_to_node.values())

        # Check that all nodes referenced by edges are contained in self.nodes,
        # and vica versa
        nodes_from_edges: Set[DTNode] = set()
        for edge in self.edges:
            nodes_from_edges.add(edge.input_node)
            nodes_from_edges.add(edge.output_node)
            assert edge.weights_initialized()

        node_set = set(self.nodes)
        for node in nodes_from_edges:
            if node not in node_set:
                assert isinstance(node, DTInputNode) or isinstance(node, DTOutputNode)

        for node in node_set:
            assert node.inon in self.inon_to_node
        for edge in self.edges:
            assert edge.inon in self.inon_to_edge

    @overrides(EXAStarGenome)
    def forward(self, input_series: TimeSeries, time_step: int) -> Dict[str, torch.Tensor]:
        """
        Performs a forward pass of the network for a given input and timestep
        Arguments:
            input_series: List of inputs
            time_step: Selected instance of the input series
        Output:
            outputs: List of output nodes and values
        """
        self.reset()
        assert sorted(self.nodes) == self.nodes

        self.input_nodes[0].forward()

        for node in filter(is_not_any_type({DTInputNode, DTOutputNode}), self.nodes):
            if node.enabled:
                if node.inputs_fired == 1:
                    val = node.get_parameter()[0]
                    x = input_series.series_dictionary[val.astype(str)][time_step]
                    node.forward(x)

        outputs = {}
        for output_node in self.output_nodes:
            outputs[output_node.node_name] = output_node.value

        return outputs

    @overrides(EXAStarGenome)
    def train_genome(
            self,
            dataset: TimeSeries,
            optimizer: torch.optim.Optimizer,
            batch_size: int,
            iterations: int,
            full: bool = False,
    ) -> float:
        """
        One round of buying and selling to determine success. Will buy and sell a given percentage of values in stocks.

        Args:
            dataset: Dataset for training
            optimizer: torch.optimizer for loss
            batch_size: desired size of days for buying and selling
            iterations: How many run through to improve the current batch.
            full: Decide to ignore batch size and train on full set
        """
        if full:
            input_series = dataset.get_inputs(dataset.input_series_names, 0)
            output_series = dataset.get_outputs_no_offset(dataset.output_series_names)
            for iteration in range(iterations + 1):
                loss = self.eval_iter(1000.0, input_series, output_series)
                if iteration < iterations:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    loss = float(loss)
            logger.info(f"final fitness (loss): {loss}, type: {type(loss)}, gen:{self.generation_number}")
            return loss
        else:
            avg = 0
            for iteration in range(iterations + 1):
                if not full:
                    rand = np.random.randint(0, len(dataset) - batch_size)
                    input_series = dataset.get_batch_inputs(dataset.input_series_names, rand, batch_size)
                    output_series = dataset.get_batch_outputs(dataset.output_series_names, rand, batch_size)
                else:
                    input_series = dataset.get_inputs(dataset.input_series_names, 0)
                    output_series = dataset.get_outputs_no_offset(dataset.output_series_names)

                loss = self.eval_iter(1000.0, input_series, output_series)
                if iteration < iterations:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                avg += float(loss)
            logger.info(f"final fitness (loss): {float(avg/iteration)}, type: {type(float(avg/iteration))}, gen:{self.generation_number}")
            return avg/iteration

    def test_genome(
            self,
            dataset: TimeSeries,
            full: bool = False,
    ) -> float:
        """
        One round of buying and selling to determine success. Will buy and sell a given percentage of values in stocks.

        Args:
            dataset: Dataset for training
            full: should the whole time be evaluated or a section
        """
        if full:
            input_series = dataset.get_batch_inputs(dataset.input_series_names, 0, 10)
            output_series = dataset.get_batch_outputs(dataset.output_series_names,0, 10)
        else:
            input_series = dataset.get_inputs(dataset.input_series_names, 0)
            output_series = dataset.get_outputs_no_offset(dataset.output_series_names)

        loss = self.eval_iter(input_series, output_series)

        return loss

    def test_genome_hist(
            self,
            dataset: TimeSeries,
            full: bool = False,
    ) -> float:
        """
        A version of test_genome that records history of trading.
        One round of buying and selling to determine success. Will buy and sell a given percentage of values in stocks.

        Args:
            dataset: Dataset for training
            full: should the whole time be evaluated or a section
        """
        if full:
            input_series = dataset.get_batch_inputs(dataset.input_series_names, 0, 10)
            output_series = dataset.get_batch_outputs(dataset.output_series_names,0, 10)
        else:
            input_series = dataset.get_inputs(dataset.input_series_names, 0)
            output_series = dataset.get_outputs_no_offset(dataset.output_series_names)

        loss, hist = self.eval_iter_hist(input_series, output_series)

        return loss, hist

    def test_genome_daily(
            self,
            dataset: TimeSeries,
    ) -> float:
        """
        Evaluate Genomes based on a day to day change, buying then selling the next day, or selling and buying the next day

        Args:
            dataset: Dataset for testing
        """

        input_series = dataset.get_inputs(dataset.input_series_names, 0)
        output_series = dataset.get_outputs_no_offset(dataset.output_series_names)

        val = 0
        b_error = 0
        s_error = 0
        b_right = 0
        s_right = 0
        print(len(input_series))
        for i in range(len(input_series)-1):
            self.reset()
            outputs = self.forward(input_series, i)
            for parameter_name, value in outputs.items():
                if parameter_name != "Hold":
                    price = output_series.series_dictionary[parameter_name][i]
                    next_p = output_series.series_dictionary[parameter_name][i + 1]
                    shift = (next_p - price)
                    if value > 0:
                        val += (shift * value)
                        if shift < 0 and value != 0:
                            b_error += 1

                    elif value < 0:
                        val += (shift * value)
                        if shift > 0 and value != 0:
                            s_error += 1
                    if shift > 0:
                        b_right += 1
                    else:
                        s_right += 1
        return val, b_error/(b_right), s_error/(s_right)

    def eval_iter_daily(self, input_series: TimeSeries, output_series: TimeSeries):
        """
            One round of buying and selling to determine success.
            Will buy and sell a given percentage of values in stocks.
            Based on a day to day change, buying then selling the next day, or selling and buying the next day

            Args:
                input_series: Input series used to pass forward in nodes
                output_series: Output series used to provide parameter information
        """
        val = torch.tensor(0.0, requires_grad=True)
        for i in range(len(input_series) - 1):
            self.reset()
            outputs = self.forward(input_series, i)
            for parameter_name, value in outputs.items():
                if parameter_name != "Hold" and value != 0:
                    price = output_series.series_dictionary[parameter_name][i]
                    next_p = output_series.series_dictionary[parameter_name][i + 1]
                    cap = price * value
                    if cap > 1000:
                        # val = val + ((next_p - price) * value)
                        value = 1000/price
                    elif cap < -1000:
                        value = -1000 / price
                        # val = val - ((next_p - price) * value)
                    val = val + (next_p * value) - cap


        if val <= 1:
            loss = -1 * val + 100
        else:
            loss = 100 / val

        return loss


    def eval_iter_single_daily(self, input_series: TimeSeries, output_series: TimeSeries):
        """
            One round of buying and selling to determine success. Will buy and sell a given percentage of values in stocks.
            Based on a day to day change, buying then selling the next day, or selling and buying the next day, only
            able to choose one share of one stock each step.

            Args:
                input_series: Input series used to pass forward in nodes
                output_series: Output series used to provide parameter information
        """
        money = torch.tensor(0.0, requires_grad=True)
        for i in range(len(input_series) - 1):
            self.reset()
            outputs = self.forward(input_series, i)
            for parameter_name, value in outputs.items():
                if parameter_name != "Hold":
                    price = output_series.series_dictionary[parameter_name][i]
                    next_p = output_series.series_dictionary[parameter_name][i + 1]
                    shift = (next_p - price)
                    if value > 0:
                        # money += (shift * value)
                        money = money + shift * value
                    elif value < 0:
                        # money += (shift * value)
                        money = money - shift * value

        if money <= 1:
            loss = -1 * money + 100
        else:
            loss = 100 / money

        return loss

    def eval_iter(self, budget: float, input_series: TimeSeries, output_series: TimeSeries):
        """
                One round of buying and selling to determine success. Will buy and sell a given percentage of values in stocks.
                Selling is one day delay only, but bought stocks carry forward.
                Args:
                    budget: set amount of starting money
                    input_series: Input series used to pass forward in nodes
                    output_series: Output series used to provide parameter information
                """
        money = torch.tensor(budget, requires_grad=True)
        held_shares = {key: torch.tensor(0.0) for key in output_series.series_dictionary}
        for i in range(len(input_series)-1):
            outputs = self.forward(input_series, i)
            for parameter_name, value in outputs.items():
                if parameter_name != "Hold":
                    if value > 0:
                        if money > 0:
                            money_to_purchase = value * money
                            price = output_series.series_dictionary[parameter_name][i]
                            bought = money_to_purchase / price
                            money = money - money_to_purchase
                            held_shares[parameter_name] = held_shares[parameter_name] + bought
                    elif value < 0:
                        price = output_series.series_dictionary[parameter_name][i]
                        money_to_sell = (-1 * value) * (money + price * held_shares[parameter_name])
                        sold_shares = money_to_sell / price
                        if held_shares[parameter_name] > sold_shares:
                            money = money + money_to_sell
                            held_shares[parameter_name] = held_shares[parameter_name] - sold_shares
                        else:
                            money = money + held_shares[parameter_name]*price
                            sold_shares = sold_shares - held_shares[parameter_name]
                            held_shares[parameter_name] = 0
                            money = money + -1 * (output_series.series_dictionary[parameter_name][i+1] - price) * sold_shares

        share_val = torch.tensor(0)
        for key in held_shares:
            share_val = share_val + held_shares[key] * output_series.series_dictionary[key][len(input_series) - 1]
        profit_func = money + share_val
        if profit_func <= 1:
            loss = -1 * profit_func + 100
        else:
            loss = 100 / profit_func

        return loss

    def eval_iter_hist(self, input_series: TimeSeries, output_series: TimeSeries):
        """
        A version of test_genome that records history of trading.
        One round of buying and selling to determine success. Will buy and sell a given percentage of values in stocks.

        Args:
            input_series: Input series used to pass forward in nodes
            output_series: Output series used to provide parameter information
        """
        val = torch.tensor(1000.0, requires_grad=True)
        held_shares_hist = {key: [] for key in output_series.series_dictionary}
        held_shares_hist["Value"] = []
        held_shares = {key: torch.tensor(0.0) for key in output_series.series_dictionary}
        for i in range(len(input_series)-1):
            outputs = self.forward(input_series, i)
            for parameter_name, value in outputs.items():

                if parameter_name != "Hold":
                    if value > 0:
                        if val > 0:
                            # print(val)
                            # print(held_shares)
                            money_to_purchase = value * val
                            price = output_series.series_dictionary[parameter_name][i]
                            bought = money_to_purchase / price
                            val = val - money_to_purchase
                            held_shares[parameter_name] = held_shares[parameter_name] + bought
                    elif value < 0:
                        price = output_series.series_dictionary[parameter_name][i]
                        money_to_sell = (-1 * value) * (money + price * held_shares[parameter_name])
                        sold_shares = money_to_sell / price
                        if held_shares[parameter_name] > sold_shares:
                            money = money + money_to_sell
                            held_shares[parameter_name] = held_shares[parameter_name] - sold_shares
                        else:
                            money = money + held_shares[parameter_name] * price
                            sold_shares = sold_shares - held_shares[parameter_name]
                            held_shares[parameter_name] = 0
                            money = money + -1 * (output_series.series_dictionary[parameter_name][i + 1] - price) * sold_shares

            # print(val, held_shares)
            share_val = 0
            for key in held_shares:
                if isinstance(held_shares[key], int):
                    held_shares_hist[key].append(held_shares[key])
                else:
                    held_shares_hist[key].append(held_shares[key].detach())
                if held_shares[key] < 0:
                    held_shares[key] = 0
                else:
                    share_val = share_val + held_shares[key] * output_series.series_dictionary[key][i]


            held_shares_hist["Value"].append(val+share_val)
        share_val = torch.tensor(0)
        for key in held_shares:
            share_val = share_val + held_shares[key] * output_series.series_dictionary[key][len(input_series) - 1]
        profit_func = val + share_val
        if profit_func <= 1:
            loss = -1 * profit_func + 100
        else:
            loss = 100 / profit_func

        return loss, held_shares_hist

    @overrides(EXAStarGenome)
    def add_edge(self, edge: DTBaseEdge) -> None:
        """
        Adds an edge when creating this gnome.

        Args:
            edge: is the edge to add
        """
        assert edge.inon not in self.inon_to_edge

        bisect.insort(self.edges, edge)
        self.inon_to_edge[edge.inon] = edge
        self.torch_modules.append(edge)

    @overrides(EXAStarGenome)
    def add_node(self, node: DTNode) -> None:
        """
        Adds an non-input and non-output node when creating this genome
        Args:
            node: is the node to add to the computational graph
        """
        assert node.inon not in self.inon_to_node

        bisect.insort(self.nodes, node)
        self.inon_to_node[node.inon] = node
        self.torch_modules.append(node)

        if isinstance(node, DTInputNode):
            bisect.insort(self.input_nodes, node)
            self.inon_to_input_node[node.inon] = node
        elif isinstance(node, DTOutputNode):
            bisect.insort(self.output_nodes, node)
            self.inon_to_output_node[node.inon] = node

    def unnormalize_node(self, node: DTNode):
        """
        Denormalizes the weight passed to the Node
        Args:
            node: is the node to denormalize
        """
        if str(node.parameter_name[0]) in self.guide.keys():
            mean, std = self.guide[str(node.parameter_name[0])]
            y = (node.input_edge.weight.item() * std) + mean

            return y
        return node.input_edge.weight.item()

    def unnormalize_edge_in(self, edge: DTBaseEdge):
        """
        Denormalizes the weight found in the edge for the input node to given edge
            Args:
                edge: is the edge to denormalize
        """
        if str(edge.input_node.parameter_name[0]) in self.guide.keys():
            mean, std = self.guide[str(edge.input_node.parameter_name[0])]
            y = (edge.input_node.input_edge.weight.item() * std) + mean

            return y
        return edge.input_node.input_edge.weight.item()

    def unnormalize_edge_out(self, edge: DTBaseEdge):
        """
        Denormalizes the weight found in the edge
            Args:
                edge: is the edge to denormalize
        """
        if str(edge.output_node.parameter_name[0]) in self.guide.keys():
            mean, std = self.guide[str(edge.output_node.parameter_name[0])]
            y = (edge.weight.item() * std) + mean
            return y
        return edge.weight.item()

    def unnormalize_para(self, parameter: str, val: float):
        """
        Denormalizes the weight found in the edge
            Args:
                edge: is the edge to denormalize
        """
        if str(parameter) in self.guide.keys():
            mean, std = self.guide[str(parameter)]
            y = (val * std) + mean
            return y
        return None
