import pandas as pd

def run(buy_sell, price):
    b_error = 0
    s_error = 0
    b_right = 0
    s_right = 0
    val = 0
    for i in range(len(buy_sell) - 1):
        shift = price[i + 1] - price[i]
        if buy_sell[i] > 0:
            val += shift
            if shift < 0:
                b_error += 1
            else:
                b_right += 1
        if buy_sell[i] < 0:
            val -= shift
            if shift > 0:
                s_error += 1
            else:
                s_right += 1

    print(val)
    print(b_error/(b_right + b_right))
    print(s_error/(s_right + s_right))
    print()
    return val

def run(buy_sell, price):
    b_error = 0
    s_error = 0
    b_right = 0
    s_right = 0
    val = 0
    for i in range(len(buy_sell) - 1):
        shift = price[i + 1] - price[i]
        if buy_sell[i] > 0:
            val += shift
            if shift < 0:
                b_error += 1
        if buy_sell[i] < 0:
            val -= shift
            if shift > 0:
                s_error += 1
        if shift > 0:
            b_right += 1
        else:
            s_right += 1

    print(val)
    print(b_error/(b_right))
    print(s_error/(s_right))
    print()
    return val

def hold_stocks():
    csv_dict = pd.read_csv("C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv", encoding="UTF-8")
    c1 = csv_dict["CPT"]
    c2 = csv_dict["UHS"]
    c3 = csv_dict["RHI"]
    c4 = csv_dict["STLD"]
    c5 = csv_dict["KMX"]
    sum = c1[0] + c2[0] + c3[0] + c4[0] + c5[0]
    shares = 1000/sum
    val = (c1[len(c1)-1] - c1[0]) + (c2[len(c2)-1] - c2[0]) + (c3[len(c3)-1] - c3[0]) + (c4[len(c4)-1] - c4[0]) + (c5[len(c5)-1] - c5[0])
    print(val*shares)


def hold_stock(name):
    csv_dict = pd.read_csv("test.csv", encoding="UTF-8")
    c1 = csv_dict[f"{name}"]
    val = (c1[len(c1) - 1] - c1[0])
    shares = 1000/c1[0]
    print(shares*val)
def main():
    """
    Quick method to test file for accuracy.
    """
    csv_dict = pd.read_csv("C:/Users/matts/Documents/RIT/exa-star-dt/src/short_data/test.csv", encoding="UTF-8")

    price = csv_dict["CPT"]
    buy_sell = csv_dict["Predicted_CPT"]
    print("CPT")
    print(price[0])
    print(f"CPT gross: {(price[len(price)-1]- price[0])/price[0]}")
    run_sum = run(buy_sell, price)
    price = csv_dict["UHS"]
    buy_sell = csv_dict["Predicted_UHS"]
    print("UHS")
    print(price[0])
    print(f"UHS gross: {(price[len(price) - 1] - price[0])/price[0]}")
    run_sum += run(buy_sell, price)
    price = csv_dict["RHI"]

    buy_sell = csv_dict["Predicted_RHI"]
    print("RHI")
    print(price[0])
    print(f"RHI gross: {(price[len(price) - 1] - price[0])/price[0]}")

    run_sum += run(buy_sell, price)
    price = csv_dict["STLD"]
    buy_sell = csv_dict["Predicted_STLD"]
    print("STLD")
    print(price[0])
    print(f"STLD gross: {(price[len(price) - 1] - price[0])/price[0]}")
    run_sum += run(buy_sell, price)
    price = csv_dict["KMX"]
    buy_sell = csv_dict["Predicted_KMX"]
    print("KMX")
    print(price[0])
    print(f"KMX gross: {(price[len(price) - 1] - price[0])/price[0]}")
    run_sum += run(buy_sell, price)
    print(run_sum)


if __name__ == "__main__":
    main()
    hold_stocks()