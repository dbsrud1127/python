"""
    Numpy 연습문제
"""

# =========== 이곳에 필요한 모듈 import 한 후 실행 ===========
import numpy as np

"""
    1. 다음 리스트 [1, 2, 3, 4, 5]를 ndarray로 변환하고, 배열의 차원(ndim)과 형태(shape)을 출력하시오.
"""
list = [1,2,3,4,5]

arr = np.array(list)

print(f"배열의 차원(ndim) :", arr.ndim)
print(f"배열의 형태(shape):", arr.shape)

"""
    2. np.arange()를 이용해 0부터 20까지의 짝수로 이루어진 배열을 생성하시오.
"""
arr = np.arange(0, 21, 2)
print(arr)

"""
    3. 다음 제시된 리스트를 배열로 변환한 후, 3 이상인 값만 추출하는 불리언 인덱싱 코드를 작성하시오.
"""
list3 = [1,3,5,2,8,3]

arr = np.array(list3)

result = arr[arr >= 3]

print(result)


"""
    4. 다음 제시된 리스트를 배열로 변환한 후, 두 번째 행만 슬라이싱하여 출력하시오.
"""
list4 = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]

arr = np.array(list4)
result = arr[1, :]
print(result)

"""
    5. 다음 제시된 리스트를 배열로 변환한 후, 모든 홀수에만 10을 더하는 벡터화 연산을 수행하시오.
"""
list5 = [1, 2, 3, 4, 5]

arr = np.array(list5)
arr[arr % 2 != 0] += 10
print(arr)

"""
    6. 다음 제시된 실수 리스트를 배열로 변환한 후, 반올림한 정수형 배열로 변환하시오.
        주의 : astype 만 쓰면 소수점 아래가 버려져서 값이 새어나간다.

        [출력 예시]
            np.array([52000.9, -3.7]) -> [52001, -4]

        [힌트] 반올림을 먼저 하고 타입을 바꾼다. 순서가 중요하다.
"""
list6 = [52000.9, 51999.2, -3.7, 52000.5]
arr = np.array(list6)

rounded_arr = np.round(list6).astype(int)

print(rounded_arr)
# =========== 아래 문제들은 실습용 데이터(prices.csv, load_utils.py)를 활용하여 풀이해보세요. =========== 
"""
    7. 첫 종목의 종가 데이터를 기준으로 최저가, 최고가와 그에 해당하는 날짜를 각각 출력하시오. 
       (실습용 데이터의 prices 와 dates 는 길이와 순서가 같다.)

        [출력 예시]
            (25899, numpy.datetime64('2023-10-16'), 8885, numpy.datetime64('2026-05-21'))

        [힌트] max 는 '값', argmax 는 '그 값이 있는 위치' 다.
              위치를 얻으면 길이가 같은 다른 배열에서 같은 자리를 꺼낼 수 있다.
"""
from load_utils import load_one_stock, load_dates

# 1. 첫 종목의 종가 데이터(prices)와 날짜 데이터(dates) 불러오기
# load_one_stock(0)은 자동으로 0번째(첫 번째) 종목의 750개 데이터를 가져옵니다.
prices = load_one_stock(0)
dates = load_dates()

# 2. 최저가와 최고가가 위치한 인덱스(방 번호) 찾기
min_idx = np.argmin(prices)  # 최저가 위치
max_idx = np.argmax(prices)  # 최고가 위치

# 3. 인덱스를 활용해 각각의 값과 날짜 매칭하기
min_val = prices[min_idx]
min_date = dates[min_idx]

max_val = prices[max_idx]
max_date = dates[max_idx]

# 4. 문제의 출력 예시 형태와 동일하게 결과 출력
# 출력 예시 순서: (최고가, 최고가 날짜, 최저가, 최저가 날짜)
result = (max_val, max_date, min_val, min_date)
print(result)

"""
    8. 종가 데이터를 기준으로 각 종목별 평균가와, 날짜별 평균가를 구하시오. 
       또한, 각 종목에서 자기 평균을 뺀 배열을 구하시오.
       
       [힌트]
       - 종목별 평균가 : (종목 수,) 배열 
       
       - 날짜별 평균가 : (날짜 수,) 배열
       
       - 각 종목에서 자기 평균을 뺀 배열 : (종목 수, 날짜 수) 배열 
         결과의 종목별 평균은 0 이 되어야 한다.
"""
from load_utils import load_matrix

# 1. (120, 750) 크기의 전체 종가 행렬 데이터 불러오기
matrix = load_matrix()

# 2. 종목별 평균가 구하기 (행 단위 평균 -> axis=1)
# 출력 크기: (120,)
stock_means = np.mean(matrix, axis=1)

# 3. 날짜별 평균가 구하기 (열 단위 평균 -> axis=0)
# 출력 크기: (750,)
date_means = np.mean(matrix, axis=0)

# 4. 각 종목에서 자기 평균을 뺀 배열 구하기 (브로드캐스팅 활용)
# (120, 750) - (120, 1) 형태로 구조를 맞춰서 빼주어야 연산이 가능합니다.
# stock_means[:, np.newaxis]를 통해 (120,)을 (120, 1)로 차원을 확장합니다.
demeaned_matrix = matrix - stock_means[:, np.newaxis]

# 5. 결과 확인 출력
print("1. 종목별 평균가 배열 크기:", stock_means.shape)
print("2. 날짜별 평균가 배열 크기:", date_means.shape)
print("3. 평균을 뺀 배열 크기:", demeaned_matrix.shape)

# 검증: 결과 배열의 종목별 평균이 0에 수렴하는지 확인 (부동소수점 오차 감안)
print("4. 첫 종목의 평균 차감 후 평균값:", np.allclose(np.mean(demeaned_matrix, axis=1), 0))