class Matrix:

    def __init__(self, rowCount, columnCount):
        self.rowCount = rowCount
        self.columnCount = columnCount
        self.values = []

        # başlangıç değeri olarak bütün değerleri 0 yapıyoruz
        for y in range(self.rowCount):
            row = []
            for x in range(self.columnCount):
                row.append(0)
            self.values.append(row)

    # yardımcı metotlar
    #####################################
    @staticmethod
    def give_error():
        print("işlem gerçekleştirilemez")  # değişecek

    @staticmethod
    def dot_product(nums1, nums2):
        if not len(nums1) == len(nums2):
            Matrix.give_error()
            # eğer dizilerin uzunluğu aynı değil ise bu işlem gerçekleştirilemez
        result = 0
        for x in range(len(nums1)):
            result += nums1[x] * nums2[x]
        return result

    def extract_column(self, columnIndex):
        column = []
        for i in range(self.rowCount):
            column.append(self.values[i][columnIndex])
        return column

    def extract_row(self, rowIndex):
        row = []
        for i in range(self.columnCount):
            row.append(self.values[rowIndex][i])
        return row

    def copy(self):
        copied = Matrix(self.rowCount, self.columnCount)
        for y in range(copied.rowCount):
            for x in range(copied.columnCount):
                copied.values[y][x] = self.values[y][x]
        return copied
    #####################################


    # matrix işlemleri
    #####################################
    def multiply(self, arg):
        # scaler çarpım
        if type(arg) == int or type(arg) == float:
            for y in range(self.rowCount):
                for x in range(self.columnCount):
                    self.values[y][x] *= arg
        # matris çarpımı
        else:
            if self.columnCount != arg.columnCount or self.rowCount != arg.rowCount:
                Matrix.give_error()
                return
                # boyutlar eşit değilse islem gerçekleştirilemez

            for y in range(self.rowCount):
                for x in range(self.columnCount):
                    self.values[y][x] *= arg.values[y][x]

    def transpose(self):
        result = Matrix(self.columnCount, self.rowCount)
        for y in range(result.rowCount):
            for x in range(result.columnCount):
                result.values[y][x] = self.values[x][y]
        return result

    def add(self, arg):
        # scaler toplama
        if type(arg) is not Matrix:
            for j in range(self.rowCount):
                for i in range(self.columnCount):
                    self.values[j][i] += arg
        else:
        # matris toplaması
            if self.columnCount != arg.columnCount or self.rowCount != arg.rowCount:
                Matrix.give_error()  # işlem için matris boyutları aynı olmalı
                return
            else:
                for j in range(self.rowCount):
                    for i in range(self.columnCount):
                        self.values[j][i] += arg.values[j][i]

    def sub(self, arg):
        if type(arg) == int or type(arg) == float:
            self.add(-arg)  # scaler çıkarma
        else:
            # matris çıkarması
            # arg matrisi işlem sonucu etkilenmesin diye kopyası çıkartılır
            a = arg.copy()
            a.multiply(-1)
            self.add(a)

    @staticmethod
    def matrix_product(m1, m2):
        if m1.columnCount != m2.rowCount:
            Matrix.give_error()
            return

        result = Matrix(m1.rowCount, m2.columnCount)
        for y in range(result.rowCount):
            for x in range(result.columnCount):
                row = m1.extract_row(y)
                column = m2.extract_column(x)
                a = Matrix.dot_product(row, column)
                result.values[y][x] = a
        return result
    ####################################

    # debug amaçlı
    ####################################
    def show(self):
        for i in range(self.rowCount):
            print(self.values[i])
        print("")
    ####################################
