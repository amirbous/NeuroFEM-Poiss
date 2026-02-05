#ifndef GEOMETRY_HPP
#define GEOMETRY_HPP


#include <vector>

#include <cstdint>


template<typename T_index, typename T_value>
struct Vertex {
    Vertex (): x(0.0f), y(0.0f), z(0.0f), 
               potential(0.0f), density(0.0f), v_id(0) {}

    Vertex (T_value x, T_value y, T_value z): 
           x(x), y(y), z(z), potential(0.0f), density(0.0f), v_id(0) {}

    Vertex (T_value x, T_value y, T_value z, T_index v_id): 
           x(x), y(y), z(z), potential(0.0f), density(0.0f), v_id(v_id) {}

    Vertex (T_value x, T_value y, T_value z, T_value potential, T_index v_id): 
           x(x), y(y), z(z), potential(potential), density(0.0f), v_id(v_id) {}

    Vertex (T_value x, T_value y, T_value z, T_value potential, T_value density, T_index v_id): 
           x(x), y(y), z(z), potential(potential), density(density), v_id(v_id) {}

    Vertex(const Vertex& other): 
           x(other.x), y(other.y), z(other.z),
           potential(other.potential), density(other.density), v_id(other.v_id) {}
    
    Vertex& operator=(const Vertex&) = default;

    bool operator<(const Vertex& other) const {
        return v_id < other.v_id;
    }

    bool operator==(const Vertex& other) const {
        return v_id == other.v_id;
    }    

    T_value x, y, z;
    T_value potential;
    T_value density;
    T_index v_id;
};


// just needed for processing the geometry! (getting faces of elements for extraction boundary vertices)
template<typename T_index>
struct Face {
    Face()
        : v1(0), v2(0), v3(0) {}
    Face(T_index v1, T_index v2, T_index v3)
        : v1(v1), v2(v2), v3(v3) {}
    
    Face(const Face& other)
        : v1(other.v1), v2(other.v2), v3(other.v3) {}

    Face& operator=(const Face&) = default;

    bool operator==(const Face& other) const {
        return v1 == other.v1 && v2 == other.v2 && v3 == other.v3;
    }

    void sort_faces() {
        if (v1 > v2) std::swap(v1, v2);
        if (v1 > v3) std::swap(v1, v3);
        if (v2 > v3) std::swap(v2, v3);
    }
    bool operator<(const Face& other) const {
        if (v1 != other.v1) return v1 < other.v1;
        if (v2 != other.v2) return v2 < other.v2;
        return v3 < other.v3;
    }
    T_index v1, v2, v3;

};


struct simple_logger {
    long transfer_duration;
    long solve_duration;
    float energy_joules;

};


template<typename T_index>
struct Edge {
    Edge()
        : v1(0), v2(0) {}
    Edge(T_index v1, T_index v2)
        : v1(v1), v2(v2) {}
    
    Edge(const Edge& other)
        : v1(other.v1), v2(other.v2) {}

    Edge& operator=(const Edge&) = default;

    bool operator==(const Edge& other) const {
        return v1 == other.v1 && v2 == other.v2;
    }

    void sort_edges() {
        if (v1 > v2) std::swap(v1, v2);
    }
    bool operator<(const Edge& other) const {
        if (v1 != other.v1) return v1 < other.v1;
        return v2 < other.v2;
    }
    T_index v1, v2;

};


template<typename T_index>
struct Element {
    
    Element()
        : v1(0), v2(0), v3(0), v4(0) {}

    Element(T_index v1, T_index v2, T_index v3, T_index v4)
        : v1(v1), v2(v2), v3(v3), v4(v4) {}

    Element(T_index v1, T_index v2, T_index v3)
        : v1(v1), v2(v2), v3(v3), v4(0) {}

    Element(const Element& other)
        : v1(other.v1), v2(other.v2), v3(other.v3), v4(other.v4) {}

    Element& operator=(const Element&) = default;

    bool operator==(const Element& other) const {
        return v1 == other.v1 && v2 == other.v2 && v3 == other.v3 && v4 == other.v4;
    }

    T_index v1, v2, v3, v4; 
};



template<typename T_index, typename T_value>
struct Model {


    int n_vertices;
    int n_elements;
    std::vector<Vertex<T_index, T_value>> vertices;
    std::vector<Element<T_index>> elements;

    Model() : n_vertices(0), n_elements(0) {}


    Model(const std::vector<Vertex<T_index, T_value>>& verts, const std::vector<Element<T_index>>& elems)
    : n_vertices(static_cast<int>(verts.size())),
      n_elements(static_cast<int>(elems.size())),
      vertices(verts),
      elements(elems)
    {}



    Model(int n_vertices, int n_elements)
        : n_vertices(n_vertices),
          n_elements(n_elements) {
        vertices.resize(n_vertices);
        elements.resize(n_elements);
    }
    
    void update_counts() {
        n_vertices = static_cast<int>(vertices.size());
        n_elements = static_cast<int>(elements.size());
    }
};


template<typename T_index, typename T_value>
struct Report {
        T_index n_vertices;
        T_index n_nonzero;
        T_value max_edge_length;
        T_value residual_norm;
        T_value energy_joules;
        long assemble_time;
        long transfer_time;
        long solve_time;

};


template<typename T_index, typename T_value>
struct CSR_matrix {
    int n_rows;
    int n_cols;
    int n_nonzero;
    std::vector<T_index> row_ptr;
    std::vector<T_index> col_ind;
    std::vector<T_value> values;

    CSR_matrix() : n_rows(0), n_cols(0), n_nonzero(0) {}

    CSR_matrix(int rows, int cols, int nonzero)
        : n_rows(rows), n_cols(cols), n_nonzero(nonzero) {
        row_ptr.resize(n_rows + 1);
        col_ind.resize(n_nonzero);
        values.resize(n_nonzero);
    }
};

#endif