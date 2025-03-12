from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

				# Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            print('='*100)
            print(product)
            current_position = state.position.get(product, 0)
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10_000  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))


          
            print(order_depth.sell_orders)
            print(order_depth.buy_orders)
            print("Current position: " + str(current_position))

            sell_size = -current_position - 20
            buy_size = -current_position + 20
            print(sell_size, buy_size)

            if product == 'AMETHYSTS':
              pass
                # Taker part takes only if ask is below 10k or bid is above 10k 
                # if list(order_depth.sell_orders.keys())[0] < 10_000:
                #   buy_order = Order('AMETHYSTS', 9_999, buy_size // 4)
                #   orders.append(buy_order)
                
                # if list(order_depth.buy_orders.keys())[0] > 10_000:
                #   sell_order = Order('AMETHYSTS', 10_001, sell_size // 4)
                #   orders.append(sell_order)
              
                # Maker part chills at +2 -2 spread from midprice
              
              buy_order = Order('AMETHYSTS', 9_998, buy_size)
              sell_order = Order('AMETHYSTS', 10_002, sell_size)
              orders.append(buy_order)
              orders.append(sell_order)
              
            if product == 'STARFRUIT':
              print("="*50)
              print("STARFRUIT")
              current_position = state.position.get(product, 0)
              print(order_depth.sell_orders)
              print(order_depth.buy_orders)
              
              ask_price = list(order_depth.sell_orders.keys())[0]
              bid_price = list(order_depth.buy_orders.keys())[0]
              midprice = (ask_price + bid_price) // 2
              
              buy_order = Order('STARFRUIT', midprice-2, buy_size)
              sell_order = Order('STARFRUIT', midprice+2, sell_size)
              orders.append(buy_order)
              orders.append(sell_order)
              
              
            result[product] = orders
    
		    # String value holding Trader state data required. 
				# It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE" 
        
				# Sample conversion request. Check more details below. 
        conversions = 1
        return result, conversions, traderData