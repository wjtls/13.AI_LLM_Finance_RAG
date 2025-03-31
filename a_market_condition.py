
import pyarrow.csv as csv
import numpy as np
import os
import pandas as pd
"""

나스닥 상승, 채권 상승, 달러 상승
경제 성장 기대와 불확실성이 공존하는 상황
주식 시장 긍정적 전망과 안전자산 선호 현상 동시 발생
미국 경제의 상대적 강세로 달러 강세



나스닥 상승, 채권 상승, 달러 하락
글로벌 경제 회복 기대감 상승
저금리 환경에서 자산 가격 전반적 상승
미국 외 국가들의 경제 회복으로 달러 약세


나스닥 상승, 채권 하락, 달러 상승
경제 성장 기대감으로 주식 시장 호조
금리 상승 예상으로 채권 가격 하락
미국 경제 강세로 달러 강세


나스닥 상승, 채권 하락, 달러 하락
글로벌 경제 회복 기대감 상승
인플레이션 우려로 채권 가격 하락
글로벌 투자 심리 개선으로 달러 약세


나스닥 하락, 채권 상승, 달러 상승
경제 불확실성 증가로 안전자산 선호
금리 하락 예상으로 채권 가격 상승
글로벌 불확실성으로 안전통화 선호


나스닥 하락, 채권 상승, 달러 하락
경기 침체 우려로 주식 시장 약세
중앙은행의 통화 완화 정책 기대로 채권 가격 상승
미국 경제 지표 악화로 달러 약세


나스닥 하락, 채권 하락, 달러 상승
글로벌 경제 위기 또는 극심한 불확실성 상황
유동성 확보를 위한 자산 매각으로 전반적 자산 가격 하락
안전통화로서 달러 선호도 증가



나스닥 하락, 채권 하락, 달러 하락
글로벌 경제 침체 우려
인플레이션 압력과 경기 둔화 우려 공존
미국 경제의 상대적 약세로 달러 약세
"""
class market_condition:
    def __init__(self):
        self.base_path = 'a_FRDdata_api/price_data_create_full_etf__adj_splitdiv'
        self.NAS = 'QQQ_full_1min_adjsplitdiv.txt'  # 나스닥 시장 ETF : QQQ
        self.bond = 'IEF_full_1min_adjsplitdiv.txt'  # 10년물 채권 추종 ETF : IEF
        self.money_index = 'UUP_full_1min_adjsplitdiv.txt'  # 달러 인덱스 : UUP

        # 데이터 로드 및 컬럼 이름 지정
        columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        try:
            self.nas_data = self._load_data(self.NAS, columns)
            self.bond_data = self._load_data(self.bond, columns)
            self.money_index_data = self._load_data(self.money_index, columns)
        except:
            self.base_path = '../a_FRDdata_api/price_data_create_full_etf__adj_splitdiv'
            self.nas_data = self._load_data(self.NAS, columns)
            self.bond_data = self._load_data(self.bond, columns)
            self.money_index_data = self._load_data(self.money_index, columns)


    def _load_data(self, filename, columns):
        file_path = os.path.join(self.base_path, filename)
        read_options = csv.ReadOptions(column_names=columns)
        parse_options = csv.ParseOptions(delimiter=',')
        convert_options = csv.ConvertOptions(
            timestamp_parsers=['%Y-%m-%d %H:%M'],
            include_columns=columns
        )

        table = csv.read_csv(file_path, read_options=read_options,
                             parse_options=parse_options,
                             convert_options=convert_options)
        df = table.to_pandas()
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    def total_market_condition_index(self, date_str, window_size, n_splits, weight_dict):

        """
        시장 상황 종합 지표
        1.변동률 계산
        2.변동률 표준화
        3.가중합 (나스닥 - (채권/달러))

        """

        target_date = pd.to_datetime(date_str)
        # 각 데이터셋에서 target_date와 가장 가까운 날짜 찾기
        nas_closest_date = self.get_closest_date(self.nas_data, target_date)
        bond_closest_date = self.get_closest_date(self.bond_data, target_date)
        money_closest_date = self.get_closest_date(self.money_index_data, target_date)

        nas_dates = self.nas_data[self.nas_data['datetime'] <= nas_closest_date]['datetime']
        nas_start_date = nas_dates.iloc[-window_size - 1]  # NAS의 시작 기준점
        # 각 지표의 변동률 계산

        nas_change = self.calculate_change_rates(self.nas_data, nas_start_date,nas_closest_date,  n_splits)
        bond_change = self.calculate_change_rates(self.bond_data, nas_start_date,bond_closest_date, n_splits)
        money_change = self.calculate_change_rates(self.money_index_data,nas_start_date, money_closest_date,  n_splits)

        # 각 지표의 변동률을 독립적으로 표준화 (평균이 0이고 표준편차가 1(데이터68%가 평균의 +-1에 분포)인 분포로 변환하여 이상치에 덜민감하고, 상대적 비교를 할수있게함)
        nas_standardized = (nas_change - np.mean(nas_change)) / np.std(nas_change)
        bond_standardized = (bond_change - np.mean(bond_change)) / np.std(bond_change)
        money_standardized = (money_change - np.mean(money_change)) / np.std(money_change)

        # 가중치를 사용하여 총 시장 상황 지표 계산
        total_indices = []
        for i in range(n_splits):
            total_index = (
                (nas_standardized[i] * weight_dict.get('nas', 0)) - (bond_standardized[i] * weight_dict.get('bond', 0) - money_standardized[i] * weight_dict.get('money', 0))
            )
            total_indices.append(str(total_index))

        return total_indices


    def calculate_change_rates(self, data, start_date, closest_date, n_splits):
        start_date = self.get_closest_date(data,start_date)
        filtered_data = data[(start_date <= data['datetime']) & (data['datetime'] <= closest_date)]
        split_size = len(filtered_data) // n_splits
        changes = []

        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(filtered_data)
            start_price = filtered_data.iloc[start_idx]['open']
            end_price = filtered_data.iloc[end_idx - 1]['open']

            #print(f'시장 상황 분할데이터 계산   시작 날짜 {filtered_data.iloc[start_idx]["datetime"]} 마지막 날짜:{filtered_data.iloc[end_idx-1]["datetime"]}')

            change = (end_price - start_price) / start_price
            changes.append(change)
        #print(f'=========================================================\n\n')
        return changes

    def get_closest_date(self, data, target_date):
        """주어진 데이터프레임에서 target_date와 가장 가까운 날짜를 찾음"""
        closest_date = data.iloc[(data['datetime'] - target_date).abs().argsort().iloc[0]]['datetime']
        return closest_date


    def calculate_window_avg(self, data, closest_date, window_size):
        """closest_date를 기준으로 window_size 만큼의 평균을 계산"""
        # closest_date를 기준으로 데이터 필터링
        filtered_data = data[(data['datetime'] <= closest_date)].tail(window_size)

        # 'value' 컬럼이 있다고 가정 (수정 필요 시 적절히 변경)
        avg_value = filtered_data['open'].mean()
        return avg_value


    def total_market_condition_index2(self, date_str, window_size, n_splits, weight_dict):

        """
        시장 상황 종합 지표
        1.변동률 계산
        2.변동률 표준화
        3.가중평균 계산 (표준화된 변동률*가중치)


        0.5 이상: 매우 긍정적인 시장 상황
        0 ~ 0.5: 다소 긍정적인 시장 상황
        -0.5 ~ 0: 다소 부정적인 시장 상황
        -0.5 이하: 매우 부정적인 시장 상황
        """

        target_date = pd.to_datetime(date_str)

        # 각 데이터셋에서 target_date와 가장 가까운 날짜 찾기
        nas_closest_date = self.get_closest_date(self.nas_data, target_date)
        bond_closest_date = self.get_closest_date(self.bond_data, target_date)
        money_closest_date = self.get_closest_date(self.money_index_data, target_date)

        # 각 지표의 변동률 계산
        nas_change = self.calculate_change_rates(self.nas_data, nas_closest_date, window_size, n_splits)
        bond_change = self.calculate_change_rates(self.bond_data, bond_closest_date, window_size, n_splits)
        money_change = self.calculate_change_rates(self.money_index_data, money_closest_date, window_size, n_splits)

        # 각 지표의 변동률을 독립적으로 표준화 (평균이 0이고 표준편차가 1(데이터68%가 평균의 +-1에 분포)인 분포로 변환하여 이상치에 덜민감하고, 상대적 비교를 할수있게함)
        nas_standardized = (nas_change - np.mean(nas_change)) / np.std(nas_change)
        bond_standardized = (bond_change - np.mean(bond_change)) / np.std(bond_change)
        money_standardized = (money_change - np.mean(money_change)) / np.std(money_change)

        # 가중치를 사용하여 총 시장 상황 지표 계산
        total_indices = []
        for i in range(n_splits):
            total_index = (
                    nas_standardized[i] * weight_dict.get('nas', 0) +
                    bond_standardized[i] * weight_dict.get('bond', 0) +
                    money_standardized[i] * weight_dict.get('money', 0)
            )
            total_indices.append(str(total_index))

        return total_indices

    def total_market_condition_index2(self, date_str, window_size, n_splits, weight_dict):
        """
        ESI = α(ΔS/σS) + β(ΔB/σB) - γ(ΔD/σD)
        ΔS, ΔB, ΔD는 주가, 채권 가격, 달러 인덱스의 변화율의 표준화값
        σS, σB, σD는 각각 주가, 채권 가격, 달러 인덱스의 표준편차
        α, β, γ는 각 요소의 가중치 (α + β + γ = 1)
        각요소 Z = (X - μ) / σ 이므로 z-score 값임 : 평균에서 얼마나 떨어졌는지 표준편차 단위로 나타냄


        ESI > 0: 경제 상황 개선 (글로벌 경제 회복 기대감)
        ESI ≈ 0: 경제 불확실성 또는 혼조세
        ESI < 0: 경제 상황 악화 (글로벌 경제 위기 우려)

        :param date_str:
        :param window_size:
        :param n_splits:
        :param weight_dict:
        :return:
        """
        target_date = pd.to_datetime(date_str)

        # 각 데이터셋에서 target_date와 가장 가까운 날짜 찾기
        nas_closest_date = self.get_closest_date(self.nas_data, target_date)
        bond_closest_date = self.get_closest_date(self.bond_data, target_date)
        money_closest_date = self.get_closest_date(self.money_index_data, target_date)


        # 각 지표의 변동률 계산
        nas_change = self.calculate_change_rates(self.nas_data, nas_closest_date, window_size, n_splits)

        # NAS의 첫 시작일 추출
        nas_dates = self.nas_data[self.nas_data['datetime'] <= nas_closest_date]['datetime']
        ref_start_date = nas_dates.iloc[-window_size - 1]  # NAS의 시작 기준점

        bond_change = self.calculate_change_rates(self.bond_data,bond_closest_date,window_size,n_splits)
        money_change = self.calculate_change_rates(self.money_index_data, money_closest_date, window_size, n_splits)

        # 각 지표의 변동률을 독립적으로 표준화 (평균이 0이고 표준편차가 1(데이터68%가 평균의 +-1에 분포)인 분포로 변환하여 이상치에 덜민감하고, 상대적 비교를 할수있게함)
        nas_change = (nas_change - np.mean(nas_change)) / np.std(nas_change)
        bond_change = (bond_change - np.mean(bond_change)) / np.std(bond_change)
        money_change = (money_change - np.mean(money_change)) / np.std(money_change)

        # 각 지표의 표준편차 계산
        nas_std = np.std(nas_change)
        bond_std = np.std(bond_change)
        money_std = np.std(money_change)

        # ESI 계산
        esi_values = []
        for i in range(n_splits):
            esi = (
                    weight_dict.get('nas', 0) * (nas_change[i] / nas_std) +
                    weight_dict.get('bond', 0) * (bond_change[i] / bond_std) -
                    weight_dict.get('money', 0) * (money_change[i] / money_std)
            )
            esi_values.append(str(esi))
        return esi_values