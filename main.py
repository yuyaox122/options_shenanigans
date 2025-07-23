import options_greeks as og

def main():
    S = 30
    K = 40
    T = 240 / 365
    r = 0.01
    sigma = 0.30
    type = 'call'
    print(og.call_delta(S, K, T, r, sigma))
    return

if __name__ == "__main__":
    main()
    